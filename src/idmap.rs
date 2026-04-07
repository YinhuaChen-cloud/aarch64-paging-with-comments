// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with identity mapping.
//! 提供使用“恒等映射”方式管理页表的功能。
//!
//! See [`IdMap`] for details on how to use it.
//! 具体使用方式见 [`IdMap`] 的文档说明。

use crate::{
    MapError, Mapping,
    descriptor::{
        Descriptor, PagingAttributes, PhysicalAddress, UpdatableDescriptor, VirtualAddress,
    },
    paging::{
        Constraints, MemoryRegion, PageTable, Translation, TranslationRegime, VaRange, deallocate,
    },
};
use core::marker::PhantomData;
use core::ptr::NonNull;

/// Identity mapping, where every virtual address is either unmapped or mapped to the identical IPA.
/// 恒等映射：每个虚拟地址要么未映射，要么映射到数值完全相同的 IPA。
///
/// 该类型描述的地址变换策略非常直接：
///
/// $$PA = VA$$
///
/// 也就是说，不做偏移、不做重定位，虚拟地址的数值会被直接当作输出物理地址使用。
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IdTranslation<A: PagingAttributes> {
    _phantom: PhantomData<A>,
}

impl<A: PagingAttributes> Default for IdTranslation<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: PagingAttributes> IdTranslation<A> {
    /// 创建一个新的恒等映射翻译策略对象。
    ///
    /// 该类型本身不保存运行时状态，只借助 `PhantomData` 关联属性类型 `A`。
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// 将虚拟地址转换为同数值的物理地址。
    ///
    /// 这是恒等映射语义的核心：输入地址是多少，输出地址就是多少。
    fn virtual_to_physical(va: VirtualAddress) -> PhysicalAddress {
        PhysicalAddress(va.0)
    }
}

impl<A: PagingAttributes> Translation<A> for IdTranslation<A> {
    fn allocate_table(&mut self) -> (NonNull<PageTable<A>>, PhysicalAddress) {
        let table = PageTable::new();

        // Physical address is the same as the virtual address because we are using identity mapping
        // everywhere.
        // 假设页表所在内存本身也处于恒等映射环境中，因此可以把分配得到的虚拟指针值直接视为
        // 对应的物理地址值。
        (table, PhysicalAddress(table.as_ptr() as usize))
    }

    unsafe fn deallocate_table(&mut self, page_table: NonNull<PageTable<A>>) {
        // SAFETY: Our caller promises that the memory was allocated by `allocate_table` on this
        // `IdTranslation` and not yet deallocated. `allocate_table` used the global allocator and
        // appropriate layout by calling `PageTable::new()`.
        // 安全性：调用方保证该页表内存确实由当前 `IdTranslation` 的 `allocate_table` 分配，
        // 且尚未被释放，因此这里可以执行与分配路径匹配的释放操作。
        unsafe {
            deallocate(page_table);
        }
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable<A>> {
        // 恒等映射下的逆向变换：页表物理地址可直接解释为可访问该页表的虚拟地址。
        NonNull::new(pa.0 as *mut PageTable<A>).expect("Got physical address 0 for pagetable")
    }
}

/// Manages a level 1 page table using identity mapping, where every virtual address is either
/// unmapped or mapped to the identical IPA.
/// 使用恒等映射管理一个一级页表：每个虚拟地址要么未映射，要么映射到数值相同的 IPA。
///
/// This assumes that identity mapping is used both for the page table being managed, and for code
/// that is managing it.
/// 这里有一个关键前提：
///
/// - 被管理页表本身所在的内存必须已经是恒等映射的；
/// - 当前执行页表管理逻辑的代码，也必须通过恒等映射访问这些页表。
///
/// 否则，源码中“把虚拟地址直接当作物理地址”以及“把物理地址直接当作虚拟指针”的逻辑都将不成立。
///
/// Mappings should be added with [`map_range`](Self::map_range) before calling
/// [`activate`](Self::activate) to start using the new page table. To make changes which may
/// require break-before-make semantics you must first call [`deactivate`](Self::deactivate) to
/// switch back to a previous static page table, and then `activate` again after making the desired
/// changes.
/// 通常应先通过 [`map_range`](Self::map_range) 建立映射，再调用 [`activate`](Self::activate)
/// 使页表生效。若后续修改可能触发 break-before-make 语义，则应先调用
/// [`deactivate`](Self::deactivate) 切回旧页表，修改完成后再重新激活。
///
/// # Example
///
/// ```no_run
/// use aarch64_paging::{
///     idmap::IdMap,
///     descriptor::El1Attributes,
///     paging::{MemoryRegion, TranslationRegime, El1And0},
/// };
///
/// const ASID: usize = 1;
/// const ROOT_LEVEL: usize = 1;
/// const NORMAL_CACHEABLE: El1Attributes = El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE);
///
/// // Create a new EL1 page table with identity mapping.
/// let mut idmap = IdMap::with_asid(ASID, ROOT_LEVEL, El1And0);
/// // Map a 2 MiB region of memory as read-write.
/// idmap.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::VALID | El1Attributes::ACCESSED,
/// ).unwrap();
/// // SAFETY: Everything the program uses is within the 2 MiB region mapped above.
/// let ttbr = unsafe {
///     // Set `TTBR0_EL1` to activate the page table.
///     idmap.activate()
/// };
///
/// // Write something to the memory...
///
/// // SAFETY: The program will only use memory within the initially mapped region until `idmap` is
/// // reactivated below.
/// unsafe {
///     // Restore `TTBR0_EL1` to its earlier value while we modify the page table.
///     idmap.deactivate(ttbr);
/// }
/// // Now change the mapping to read-only and executable.
/// idmap.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::READ_ONLY | El1Attributes::VALID
///     | El1Attributes::ACCESSED,
/// ).unwrap();
/// // SAFETY: Everything the program will used is mapped in by this page table.
/// unsafe {
///     idmap.activate();
/// }
/// ```
#[derive(Debug)]
pub struct IdMap<R: TranslationRegime> {
    mapping: Mapping<IdTranslation<R::Attributes>, R>,
}

impl<R: TranslationRegime<Asid = (), VaRange = ()>> IdMap<R> {
    /// Creates a new identity-mapping page table with the given root level.
    /// 创建一个使用恒等映射的页表，并指定根页表级别。
    pub fn new(rootlevel: usize, regime: R) -> Self {
        Self {
            mapping: Mapping::new(IdTranslation::<R::Attributes>::new(), rootlevel, regime),
        }
    }
}

impl<R: TranslationRegime<Asid = usize, VaRange = VaRange>> IdMap<R> {
    /// Creates a new identity-mapping page table with the given ASID and root level.
    /// 创建一个使用恒等映射的页表，并指定 ASID、根页表级别以及翻译体制。
    pub fn with_asid(asid: usize, rootlevel: usize, regime: R) -> Self {
        Self {
            mapping: Mapping::with_asid_and_va_range(
                IdTranslation::<R::Attributes>::new(),
                asid,
                rootlevel,
                regime,
                VaRange::Lower,
            ),
        }
    }
}

impl<R: TranslationRegime> IdMap<R> {
    /// Returns the size in bytes of the virtual address space which can be mapped in this page
    /// table.
    /// 返回当前页表能够覆盖的虚拟地址空间大小（单位：字节）。
    ///
    /// This is a function of the chosen root level.
    /// 其大小由所选根页表级别决定。
    pub fn size(&self) -> usize {
        self.mapping.size()
    }

    /// Activates the page table by programming the physical address of the root page table into
    /// `TTBRn_ELx`, along with the provided ASID. The previous value of `TTBRn_ELx` is returned so
    /// that it may later be restored by passing it to [`deactivate`](Self::deactivate).
    /// 通过把根页表物理地址以及对应 ASID 写入 `TTBRn_ELx` 来激活当前页表。
    /// 返回值是激活前 `TTBRn_ELx` 的旧值，之后可传给 [`deactivate`](Self::deactivate)
    /// 用于恢复。
    ///
    /// In test builds or builds that do not target aarch64, the `TTBR0_EL1` access is omitted.
    /// 在测试构建或非 aarch64 目标平台上，不会真正访问系统寄存器。
    ///
    /// # Safety
    ///
    /// The caller must ensure that the page table doesn't unmap any memory which the program is
    /// using, or introduce aliases which break Rust's aliasing rules. The page table must not be
    /// dropped as long as its mappings are required, as it will automatically be deactivated when
    /// it is dropped.
    /// 调用者必须保证：
    ///
    /// - 激活后不会把程序当前仍在使用的内存取消映射；
    /// - 不会引入破坏 Rust 别名规则的映射别名；
    /// - 在这些映射仍有用时，页表对象本身不能被提前释放。
    pub unsafe fn activate(&mut self) -> usize {
        // SAFETY: We delegate the safety requirements to our caller.
        unsafe { self.mapping.activate() }
    }

    /// Deactivates the page table, by setting `TTBRn_ELx` to the provided value, and invalidating
    /// the TLB for this page table's configured ASID. The provided TTBR value should be the value
    /// returned by the preceding [`activate`](Self::activate) call.
    /// 通过恢复旧 `TTBRn_ELx` 值并对该页表的 ASID 执行 TLB 失效，来停用当前页表。
    ///
    /// In test builds or builds that do not target aarch64, the `TTBR0_EL1` access is omitted.
    /// 在测试构建或非 aarch64 平台上，不会真正访问系统寄存器。
    ///
    /// # Safety
    ///
    /// The caller must ensure that the previous page table which this is switching back to doesn't
    /// unmap any memory which the program is using.
    /// 调用者必须保证：切回去的旧页表仍然映射着程序继续执行所需的全部内存。
    pub unsafe fn deactivate(&mut self, previous_ttbr: usize) {
        // SAFETY: We delegate the safety requirements to our caller.
        unsafe {
            self.mapping.deactivate(previous_ttbr);
        }
    }

    /// Maps the given range of virtual addresses to the identical physical addresses with the given
    /// flags.
    /// 将给定虚拟地址范围映射到“同数值”的物理地址范围，并写入给定属性标志。
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active. This function writes block and page entries, but only maps them if `flags`
    /// contains [`PagingAttributes::VALID`], otherwise the entries remain invalid.
    /// 一般应在页表未激活时调用。特别是，任何可能触发架构上 break-before-make 约束的修改，
    /// 都必须在页表不活跃时进行。
    ///
    /// 对于此前未映射的地址范围，在页表活跃时新增映射通常是允许的。
    ///
    /// 本函数会写入 block/page 描述符；但只有当 `flags` 含有
    /// [`PagingAttributes::VALID`] 时，这些描述符才会成为有效映射，否则条目仍保持无效状态。
    ///
    /// # Errors
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    /// 如果地址区间起止反向，返回 [`MapError::RegionBackwards`]。
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    /// 如果范围超出当前页表所能覆盖的虚拟地址空间，返回 [`MapError::AddressRange`]。
    ///
    /// Returns [`MapError::InvalidFlags`] if the `flags` argument has unsupported attributes set.
    /// 如果 `flags` 包含不受支持的属性，返回 [`MapError::InvalidFlags`]。
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation`] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    /// 如果对 live 映射的修改违反 break-before-make（BBM）规则，返回
    /// [`MapError::BreakBeforeMakeViolation`]。
    pub fn map_range(
        &mut self,
        range: &MemoryRegion,
        flags: R::Attributes,
    ) -> Result<(), MapError> {
        self.map_range_with_constraints(range, flags, Constraints::empty())
    }

    /// Maps the given range of virtual addresses to the identical physical addresses with the given
    /// given flags, taking the given constraints into account.
    /// 在考虑额外约束 `constraints` 的前提下，将给定虚拟地址范围映射到同数值的物理地址范围。
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active. This function writes block and page entries, but only maps them if `flags`
    /// contains [`PagingAttributes::VALID`], otherwise the entries remain invalid.
    /// 调用时机和错误条件与 [`map_range`](Self::map_range) 基本相同；区别在于这里允许显式
    /// 指定附加限制，例如禁止使用 block 映射。
    ///
    /// # Errors
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    ///
    /// Returns [`MapError::InvalidFlags`] if the `flags` argument has unsupported attributes set.
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation`] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn map_range_with_constraints(
        &mut self,
        range: &MemoryRegion,
        flags: R::Attributes,
        constraints: Constraints,
    ) -> Result<(), MapError> {
        // 恒等映射的关键步骤：把起始虚拟地址直接作为起始物理地址，随后交由底层通用
        // `Mapping` 完成整段映射。
        let pa = IdTranslation::<R::Attributes>::virtual_to_physical(range.start());
        self.mapping.map_range(range, pa, flags, constraints)
    }

    /// Applies the provided updater function to the page table descriptors covering a given
    /// memory range.
    /// 对覆盖给定内存范围的页表描述符应用一个更新函数。
    ///
    /// This may involve splitting block entries if the provided range is not currently mapped
    /// down to its precise boundaries. For visiting all the descriptors covering a memory range
    /// without potential splitting (and no descriptor updates), use
    /// [`walk_range`](Self::walk_range) instead.
    /// 如果给定范围与当前映射边界不完全对齐，内部可能会先把较大的 block entry 拆分成更细
    /// 粒度的条目，再调用更新函数。若只想遍历而不修改，也不希望触发拆分，请使用
    /// [`walk_range`](Self::walk_range)。
    ///
    /// The updater function receives the following arguments:
    /// 更新函数将收到以下参数：
    ///
    /// - The virtual address range mapped by each page table descriptor. A new descriptor will
    ///   have been allocated before the invocation of the updater function if a page table split
    ///   was needed.
    /// - 当前描述符所覆盖的虚拟地址范围；若需要拆分，则拆分所需的描述符已在回调前准备好。
    /// - A mutable reference to the page table descriptor that permits modifications.
    /// - 可修改的页表描述符引用。
    /// - The level of a translation table the descriptor belongs to.
    /// - 描述符所属的页表层级。
    ///
    /// The updater function should return:
    /// 更新函数的返回值约定如下：
    ///
    /// - `Ok` to continue updating the remaining entries.
    /// - 返回 `Ok`：继续处理剩余条目。
    /// - `Err` to signal an error and stop updating the remaining entries.
    /// - 返回 `Err`：停止处理并向上传播错误。
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active.
    /// 与映射接口类似，任何可能触发 break-before-make 的修改通常都应在页表不活跃时进行。
    ///
    /// # Errors
    ///
    /// Returns [`MapError::PteUpdateFault`] if the updater function returns an error.
    /// 如果更新函数返回错误，会包装成 [`MapError::PteUpdateFault`]。
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    /// 如果范围起止反向，返回 [`MapError::RegionBackwards`]。
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    /// 如果范围超出页表支持的虚拟地址空间，返回 [`MapError::AddressRange`]。
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation`] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    /// 如果修改会违反 live 映射上的 BBM 规则，返回
    /// [`MapError::BreakBeforeMakeViolation`]。
    pub fn modify_range<F>(&mut self, range: &MemoryRegion, f: &F) -> Result<(), MapError>
    where
        F: Fn(&MemoryRegion, &mut UpdatableDescriptor<'_, R::Attributes>) -> Result<(), ()>
            + ?Sized,
    {
        self.mapping.modify_range(range, f)
    }

    /// Applies the provided callback function to the page table descriptors covering a given
    /// memory range.
    /// 对覆盖给定内存范围的页表描述符执行只读遍历回调。
    ///
    /// The callback function receives the following arguments:
    /// 回调函数将收到以下参数：
    ///
    /// - The range covered by the current step in the walk. This is always a subrange of `range`
    ///   even when the descriptor covers a region that exceeds it.
    /// - 当前遍历步所对应的地址范围。即便描述符覆盖更大的区域，这里传入的也始终只是目标
    ///   `range` 的一个子区间。
    /// - The page table descriptor itself.
    /// - 当前页表描述符。
    /// - The level of a translation table the descriptor belongs to.
    /// - 描述符所属层级。
    ///
    /// The callback function should return:
    /// 回调返回值语义如下：
    ///
    /// - `Ok` to continue visiting the remaining entries.
    /// - 返回 `Ok`：继续遍历。
    /// - `Err` to signal an error and stop visiting the remaining entries.
    /// - 返回 `Err`：停止遍历。
    ///
    /// # Errors
    ///
    /// Returns [`MapError::PteUpdateFault`] if the callback function returns an error.
    /// 如果回调返回错误，则封装成 [`MapError::PteUpdateFault`]。
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    /// 如果范围反向，返回 [`MapError::RegionBackwards`]。
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    /// 如果范围超出页表可覆盖的虚拟地址空间，返回 [`MapError::AddressRange`]。
    pub fn walk_range<F>(&self, range: &MemoryRegion, f: &mut F) -> Result<(), MapError>
    where
        F: FnMut(&MemoryRegion, &Descriptor<R::Attributes>, usize) -> Result<(), ()>,
    {
        self.mapping.walk_range(range, f)
    }

    /// Looks for subtables whose entries are all empty and replaces them with a single empty entry,
    /// freeing the subtable.
    /// 查找所有“内容全为空”的子页表，并用一个空条目替代它们，同时释放对应子表内存。
    ///
    /// This requires walking the whole hierarchy of pagetables, so you may not want to call it
    /// every time a region is unmapped. You could instead call it when the system is under memory
    /// pressure.
    /// 该操作需要遍历整棵页表层级树，成本相对较高，因此通常不建议每次解除映射后立即执行；
    /// 更适合在系统内存紧张、需要回收页表开销时统一调用。
    pub fn compact_subtables(&mut self) {
        self.mapping.compact_subtables();
    }

    /// Returns the physical address of the root table.
    /// 返回根页表的物理地址。
    ///
    /// This may be used to activate the page table by setting the appropriate TTBRn_ELx if you wish
    /// to do so yourself rather than by calling [`activate`](Self::activate). Make sure to call
    /// [`mark_active`](Self::mark_active) after doing so.
    /// 如果调用者希望自己设置 `TTBRn_ELx`，而不是直接调用 [`activate`](Self::activate)，
    /// 可以先通过本函数取得根页表物理地址；手动激活后应调用 [`mark_active`](Self::mark_active)
    /// 同步库内部状态。
    pub fn root_address(&self) -> PhysicalAddress {
        self.mapping.root_address()
    }

    /// Marks the page table as active.
    /// 将当前页表标记为“活跃中”。
    ///
    /// This should be called if the page table is manually activated by calling
    /// [`root_address`](Self::root_address) and setting some TTBR with it. This will cause
    /// [`map_range`](Self::map_range) and [`modify_range`](Self::modify_range) to perform extra
    /// checks to avoid violating break-before-make requirements.
    /// 当页表通过手动写 TTBR 的方式被激活时，库本身无法自动感知；调用此函数后，后续
    /// [`map_range`](Self::map_range) 和 [`modify_range`](Self::modify_range) 会把当前页表视为
    /// live，从而执行额外检查，避免违反 break-before-make 规则。
    ///
    /// It is called automatically by [`activate`](Self::activate).
    /// 若使用 [`activate`](Self::activate)，则本函数会被自动调用。
    pub fn mark_active(&mut self) {
        self.mapping.mark_active();
    }

    /// Marks the page table as inactive.
    /// 将当前页表标记为“非活跃”。
    ///
    /// This may be called after manually disabling the use of the page table, such as by setting
    /// the relevant TTBR to a different address.
    /// 如果调用者手动切换了 TTBR，使当前页表不再被使用，应调用本函数同步库内部状态。
    ///
    /// It is called automatically by [`deactivate`](Self::deactivate).
    /// 若使用 [`deactivate`](Self::deactivate)，则本函数会被自动调用。
    pub fn mark_inactive(&mut self) {
        self.mapping.mark_inactive();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::El1Attributes;
    use crate::paging::El1And0;
    use crate::{
        MapError, VirtualAddress,
        paging::{BITS_PER_LEVEL, MemoryRegion, PAGE_SIZE},
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;
    const DEVICE_NGNRE: El1Attributes = El1Attributes::ATTRIBUTE_INDEX_0;
    const NORMAL_CACHEABLE: El1Attributes =
        El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE);

    #[test]
    fn map_valid() {
        // 这个测试函数用于验证：
        // 在 root level = 1 的恒等映射页表中，只要给出的 MemoryRegion 处于合法虚拟地址范围内，
        // `map_range()` 就应该能够成功建立映射并返回 Ok(())。
        //
        // 它覆盖了多个典型场景：
        // 1. 地址空间起始位置的极小范围（1 字节）
        // 2. 地址空间起始位置的多页范围（2 页）
        // 3. 地址空间末尾的极小范围（1 字节）
        // 4. 跨越两个子页表边界的范围
        // 5. 整个 root level 1 可覆盖的完整地址空间
        //
        // 这些场景共同验证：
        // - 合法区间不会被误判为越界；
        // - 极小区间和大区间都能正确处理；
        // - 页表边界/子表边界处的拆分与递归逻辑正确；
        // - 最大合法地址附近的边界判断正确。

        // -------- 场景 1：映射地址空间起始处的 1 字节 --------
        // 目标：验证最小非空区间 [0, 1) 可以成功映射。
        // 这能检查起始地址为 0 时的处理，以及“小于一页的范围”是否会被正确接受。
        // asid = 1: 这套页表属于 ASID = 1 的地址空间。
        // rootlevel = 1: 表示这棵页表从 L1 开始，能覆盖 2^39 字节（512 GiB）虚拟地址空间
        // regime = El1And0: 这套页表适用于 EL1 和 EL0 的地址变换。
        let mut idmap = IdMap::with_asid(1, 1, El1And0);

        // 在测试环境里，这里不会真的切换硬件页表，
        // 而是把该页表标记为 active，以便触发更接近真实环境的检查逻辑。
        let ttbr = unsafe { idmap.activate() };

        // 期望：对合法范围调用 map_range() 成功。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        // 将页表重新标记为 inactive，恢复测试前状态。
        unsafe {
            idmap.deactivate(ttbr);
        }

        // -------- 场景 2：映射地址空间起始处的 2 页 --------
        // 目标：验证连续多页范围 [0, 2 * PAGE_SIZE) 能正常映射。
        // 这覆盖“标准页粒度多页映射”的常规路径。
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }

        // -------- 场景 3：映射地址空间末尾的 1 字节 --------
        // 目标：验证靠近最大合法地址上界的最后一个字节区间仍然是“合法范围”。
        // 这里测试的是地址上界判断是否正确，避免把 [MAX-1, MAX) 错误判成越界。
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }

        // -------- 场景 4：映射跨越两个子表边界的两页 --------
        // 目标：验证当范围横跨页表下一级子表边界时，内部拆分和遍历逻辑仍然正确。
        // [PAGE_SIZE * 1023, PAGE_SIZE * 1025) 恰好覆盖边界前后一共两页。
        // 这是页表实现里常见的边界测试点。
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(PAGE_SIZE * 1023, PAGE_SIZE * 1025),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }

        // -------- 场景 5：映射整个有效地址空间 --------
        // 目标：验证从 0 到 root level 1 可覆盖上界的整个合法虚拟地址空间，
        // 都可以被完整映射而不报错。
        // 这覆盖了最大规模映射路径，可检查整体递归建表/填表流程是否稳定正确。
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn map_break_before_make() {
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // 先强制使用页映射而非 block 映射，便于后续验证“已按页映射时允许更细粒度修改”的情况。
        idmap
            .map_range_with_constraints(
                &MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::NO_BLOCK_MAPPINGS,
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Splitting a range is permitted if it was mapped down to pages
        // 若原先已经细化到页级映射，则进一步局部修改是允许的。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            ),
            Ok(())
        );
        unsafe {
            idmap.deactivate(ttbr);
        }

        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        idmap
            .map_range(
                &MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .ok();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Extending a range is fine even if there are block mappings
        // in the middle
        // 即使中间包含 block 映射，只要是向外扩展现有范围，也可能是允许的。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(BLOCK_SIZE - PAGE_SIZE, 2 * BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            ),
            Ok(())
        );

        // Remapping a region that intersects a block mapping is permitted if it does not result in
        // a split.
        // 与 block 映射相交的重映射，只要不导致拆分，也可以是合法的。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            ),
            Ok(())
        );

        // Splitting a range is not permitted
        // 若修改会迫使 live block 拆分，则违反 BBM 规则。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::READ_ONLY,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );

        // Partially remapping a live range read-only is only permitted
        // if it does not require splitting
        // live 范围的局部重映射只有在“不需要拆分”的前提下才允许。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::READ_ONLY,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, BLOCK_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::READ_ONLY,
            ),
            Ok(())
        );

        // Changing the memory type is not permitted
        // 修改已生效映射的内存类型属于不允许的 live 修改。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, BLOCK_SIZE),
                DEVICE_NGNRE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::NON_GLOBAL,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                0, PAGE_SIZE
            )))
        );

        // Making a range invalid is only permitted if it does not require splitting
        // 将映射变为无效，同样只有在不需要拆分时才允许。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(PAGE_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );
        assert_eq!(
            idmap.map_range(&MemoryRegion::new(PAGE_SIZE, BLOCK_SIZE), NORMAL_CACHEABLE),
            Ok(())
        );

        // Creating a new valid entry is always permitted
        // 新建原本不存在的有效映射始终允许。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, 2 * PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            ),
            Ok(())
        );

        // Setting the non-global attribute is permitted
        // 给现有映射增加 non-global 属性是允许的。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::NON_GLOBAL,
            ),
            Ok(())
        );

        // Removing the non-global attribute from a live mapping is not permitted
        // 从 live 映射中去掉 non-global 属性则不被允许。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                0, PAGE_SIZE
            )))
        );

        // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it as
        // inactive for the sake of BBM rules.
        unsafe {
            idmap.deactivate(ttbr);
        }
        // Removing the non-global attribute from an inactive mapping is permitted
        // 若页表已不活跃，则去掉 non-global 属性是允许的。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);

        // One byte, just past the edge of the valid range.
        // 超出合法虚拟地址范围 1 字节，应报 AddressRange。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1,
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );

        // From 0 to just past the valid range.
        // 从 0 一直映射到越界位置，同样应报 AddressRange。
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );
    }

    #[test]
    #[should_panic]
    fn split_live_block_mapping() -> () {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // 该测试验证：当 block 映射已经 live 时，后续若修改需要拆分该 block，
        // 最终会触发 break-before-make 相关异常路径。
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE << BITS_PER_LEVEL),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::READ_ONLY
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::READ_ONLY
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
            )
            .unwrap();
        let r = idmap.map_range(
            &MemoryRegion::new(PAGE_SIZE, 2 * PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
        );
        unsafe { idmap.deactivate(ttbr) };
        r.unwrap();
    }

    fn make_map() -> (IdMap<El1And0>, usize) {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // 建立一个简单恒等映射样例，便于后续测试 `modify_range` 相关行为。
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::READ_ONLY
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        (idmap, ttbr)
    }

    #[test]
    fn update_backwards_range() {
        let (mut idmap, ttbr) = make_map();
        assert!(
            idmap
                .modify_range(&MemoryRegion::new(PAGE_SIZE * 2, 1), &|_range, entry| {
                    entry.modify_flags(
                        El1Attributes::SWFLAG_0,
                        El1Attributes::from_bits(0usize).unwrap(),
                    )
                },)
                .is_err()
        );

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn update_range() {
        let (mut idmap, ttbr) = make_map();
        assert!(
            idmap
                .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry| {
                    if !entry.is_table() {
                        entry.modify_flags(El1Attributes::SWFLAG_0, El1Attributes::NON_GLOBAL)?;
                    }
                    Ok(())
                })
                .is_err()
        );
        idmap
            .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry| {
                if !entry.is_table() {
                    entry.modify_flags(
                        El1Attributes::SWFLAG_0,
                        El1Attributes::from_bits(0usize).unwrap(),
                    )?;
                }
                Ok(())
            })
            .unwrap();
        idmap
            .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|range, entry| {
                if !entry.is_table() {
                    assert!(entry.flags().contains(El1Attributes::SWFLAG_0));
                    assert_eq!(range.end() - range.start(), PAGE_SIZE);
                }
                Ok(())
            })
            .unwrap();
        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn breakup_invalid_block() {
        const BLOCK_RANGE: usize = 0x200000;
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        // 先建立一个带软件标志的无效 block，再把第一页单独映射成有效页，
        // 用于验证拆分后各页属性是否符合预期。
        idmap
            .map_range(
                &MemoryRegion::new(0, BLOCK_RANGE),
                NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::SWFLAG_0,
            )
            .unwrap();
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
            )
            .unwrap();
        idmap
            .modify_range(&MemoryRegion::new(0, BLOCK_RANGE), &|range, entry| {
                if entry.level() == 3 {
                    let has_swflag = entry.flags().contains(El1Attributes::SWFLAG_0);
                    let is_first_page = range.start().0 == 0usize;
                    assert!(has_swflag != is_first_page);
                }
                Ok(())
            })
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn unmap_subtable() {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map one page, which will cause subtables to be split out.
        // 先映射一页，迫使中间层子表被创建出来。
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        // Unmap the whole table's worth of address space.
        // 再把整张子表覆盖的地址范围全部取消映射。
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 512 * 512),
                El1Attributes::empty(),
            )
            .unwrap();
        // All entries in the top-level table should be 0.
        // 此时顶层表中的相关条目应当全部恢复为 0。
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn unmap_subtable_higher() {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        const ROOT_GRANULARITY: usize = PAGE_SIZE * 512 * 512;
        // Map one page in the the second entry of the root table.
        // 在根表第二个条目对应的地址区间内映射一页。
        idmap
            .map_range(
                &MemoryRegion::new(ROOT_GRANULARITY, ROOT_GRANULARITY + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        // Unmap the second entry of the root table.
        // 再把该根表条目覆盖的整个地址区间全部取消映射。
        idmap
            .map_range(
                &MemoryRegion::new(ROOT_GRANULARITY, ROOT_GRANULARITY * 2),
                El1Attributes::empty(),
            )
            .unwrap();
        // All entries in the top-level table should be 0.
        // 预期顶层表对应条目都被清空。
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn compact() {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map two pages, which will cause subtables to be split out.
        // 先映射两页，触发子表创建。
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        // Unmap the pages again.
        // 再取消映射这两页。
        idmap
            .map_range(&MemoryRegion::new(0, PAGE_SIZE * 2), El1Attributes::empty())
            .unwrap();
        // Compact to remove the subtables.
        // 显式压缩子表，把已经空掉的中间页表回收掉。
        idmap.compact_subtables();
        // All entries in the top-level table should be 0.
        // 压缩后顶层表应回到全空状态。
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn compact_blocks() {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map two blocks at level 2, which will cause subtables to be split out.
        // 先映射两个二级 block，触发中间结构建立。
        const BLOCK_SIZE: usize = PAGE_SIZE * 512;
        idmap
            .map_range(
                &MemoryRegion::new(0, BLOCK_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        // Unmap the blocks again.
        // 再取消这两个 block 映射。
        idmap
            .map_range(
                &MemoryRegion::new(0, BLOCK_SIZE * 2),
                El1Attributes::empty(),
            )
            .unwrap();
        // Compact to remove the subtables.
        // 压缩后应回收无用的子表。
        idmap.compact_subtables();
        // All entries in the top-level table should be 0.
        // 顶层表应恢复为空。
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    /// When an unmapped entry is split into a table, all entries should be zero.
    /// 当一个未映射条目被拆分成子表时，子表中的所有条目都应保持全 0。
    #[test]
    fn split_table_zero() {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);

        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        idmap
            .walk_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 20),
                &mut |_, descriptor, _| {
                    assert!(!descriptor.is_valid());
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(descriptor.flags(), El1Attributes::empty());
                    assert_eq!(descriptor.output_address(), PhysicalAddress(0));
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn modify_unmap_compact() {
        let mut idmap = IdMap::with_asid(1, 1, El1And0);
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map two pages, which will cause subtables to be split out.
        // 先映射两页，触发子表展开。
        idmap
            .map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        // Use `modify_range` to unmap the pages.
        // 使用 `modify_range` 直接把这些页改成无效映射。
        idmap
            .modify_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                &|_, descriptor| descriptor.set(PhysicalAddress(0), El1Attributes::empty()),
            )
            .unwrap();
        // Compact to remove the subtables.
        // 再压缩页表，回收已经空掉的子表。
        idmap.compact_subtables();
        // All entries in the top-level table should be 0.
        // 顶层表最终应恢复为空。
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn table_sizes() {
        // 根级别越深，单张根表覆盖的虚拟地址空间越小。
        assert_eq!(IdMap::<El1And0>::with_asid(1, 0, El1And0).size(), 1 << 48);
        assert_eq!(IdMap::<El1And0>::with_asid(1, 1, El1And0).size(), 1 << 39);
        assert_eq!(IdMap::<El1And0>::with_asid(1, 2, El1And0).size(), 1 << 30);
        assert_eq!(IdMap::<El1And0>::with_asid(1, 3, El1And0).size(), 1 << 21);
    }

    #[test]
    fn dont_use_l0_block_mapping() {
        // We don't currently support FEAT_LPA2; test that the mappings do not attempt to use huge tables
        // 当前尚不支持 FEAT_LPA2，因此即使范围足够大，也不应尝试使用 L0 巨块映射。
        let mut idmap = IdMap::with_asid(1, 0, El1And0);
        let block_size = PAGE_SIZE * 512 * 512 * 512; // 512 GiB, corresponding to Level 0
        let range = MemoryRegion::new(0, block_size);

        // The range should map at Level 1 even though it's big enough for Level 0
        // 即使范围足以覆盖 L0 block，也应从 L1 开始建立映射。
        idmap
            .map_range(
                &range,
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        assert_eq!(idmap.mapping.root.mapping_level(range.start()), Some(1));

        // The subtable should be cleaned up correctly after unmapping
        // 取消映射后，相关子表也应被正确回收。
        idmap.map_range(&range, El1Attributes::empty()).unwrap();
        idmap.compact_subtables();
        assert_eq!(idmap.mapping.root.mapping_level(range.start()), None);
    }
}
