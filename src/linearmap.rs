// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with linear mapping.
//! 提供使用“线性映射”方式管理页表的功能。
//!
//! See [`LinearMap`] for details on how to use it.
//! 具体使用方式见 [`LinearMap`] 的文档说明。

use crate::{
    MapError, Mapping,
    descriptor::{
        Descriptor, PagingAttributes, PhysicalAddress, UpdatableDescriptor, VirtualAddress,
    },
    paging::{
        Constraints, MemoryRegion, PAGE_SIZE, PageTable, Translation, TranslationRegime, VaRange,
        deallocate, is_aligned,
    },
};
use core::marker::PhantomData;
use core::ptr::NonNull;

/// Linear mapping, where every virtual address is either unmapped or mapped to an IPA with a fixed
/// offset.
/// 线性映射：每个虚拟地址要么未映射，要么映射到一个与其相差固定偏移量的
/// IPA（中间物理地址）。
///
/// 该类型本身描述的是一种地址变换策略：
///
/// $$PA = VA + offset$$
///
/// 其中 `offset` 在整个映射生命周期中保持不变。
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LinearTranslation<A: PagingAttributes> {
    /// The offset from a virtual address to the corresponding (intermediate) physical address.
    /// 从虚拟地址到对应（中间）物理地址的固定偏移量。
    offset: isize,
    _phantom: PhantomData<A>,
}

impl<A: PagingAttributes> LinearTranslation<A> {
    /// Constructs a new linear translation, which will map a virtual address `va` to the
    /// (intermediate) physical address `va + offset`.
    /// 构造一个新的线性映射策略，使任意虚拟地址 `va` 被映射到
    /// `va + offset` 对应的（中间）物理地址。
    ///
    /// The `offset` must be a multiple of [`PAGE_SIZE`]; if not this will panic.
    /// `offset` 必须是 [`PAGE_SIZE`] 的整数倍，否则会直接 panic。
    ///
    /// 这样做的原因是页表映射至少需要满足页粒度对齐；如果偏移量本身不是页大小的整数倍，
    /// 那么同一个页内的地址会被映射到不正确的物理页边界，无法形成合法的页映射关系。
    pub fn new(offset: isize) -> Self {
        if !is_aligned(offset.unsigned_abs(), PAGE_SIZE) {
            panic!(
                "Invalid offset {}, must be a multiple of page size {}.",
                offset, PAGE_SIZE,
            );
        }
        Self {
            offset,
            _phantom: PhantomData,
        }
    }

    fn virtual_to_physical(&self, va: VirtualAddress) -> Result<PhysicalAddress, MapError> {
        // 线性映射的正向变换：PA = VA + offset。
        // 这里使用带检查的加法，避免：
        // 1. 有符号加法溢出；
        // 2. 结果为负数，无法表示为无符号物理地址；
        // 3. 转换到 usize 时失败。
        if let Some(pa) = checked_add_to_unsigned(va.0 as isize, self.offset) {
            Ok(PhysicalAddress(pa))
        } else {
            Err(MapError::InvalidVirtualAddress(va))
        }
    }
}

impl<A: PagingAttributes> Translation<A> for LinearTranslation<A> {
    fn allocate_table(&mut self) -> (NonNull<PageTable<A>>, PhysicalAddress) {
        let table = PageTable::new();
        // Assume that the same linear mapping is used everywhere.
        // 假设页表自身所在内存，以及当前管理页表的代码，都运行在同一套线性映射规则下。
        // 因此可以先把分配得到的指针解释成一个虚拟地址，再按照当前 offset 计算出其物理地址。
        let va = VirtualAddress(table.as_ptr() as usize);

        let pa = self.virtual_to_physical(va).expect(
            "Allocated subtable with virtual address which doesn't correspond to any physical address."
        );
        (table, pa)
    }

    unsafe fn deallocate_table(&mut self, page_table: NonNull<PageTable<A>>) {
        // SAFETY: Our caller promises that the memory was allocated by `allocate_table` on this
        // `LinearTranslation` and not yet deallocated. `allocate_table` used the global allocator
        // and appropriate layout by calling `PageTable::new()`.
        // 安全性：调用方保证该页表内存确实由当前 `LinearTranslation` 的 `allocate_table`
        // 分配，且尚未释放；因此这里可以对其执行与分配路径匹配的回收逻辑。
        unsafe {
            deallocate(page_table);
        }
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable<A>> {
        // 线性映射的逆向变换：VA = PA - offset。
        // 该函数主要在“已知页表物理地址，反查可访问它的虚拟地址”时使用。
        let signed_pa = pa.0 as isize;
        if signed_pa < 0 {
            panic!("Invalid physical address {} for pagetable", pa);
        }
        if let Some(va) = signed_pa.checked_sub(self.offset) {
            if let Some(ptr) = NonNull::new(va as *mut PageTable<A>) {
                ptr
            } else {
                panic!(
                    "Invalid physical address {} for pagetable (translated to virtual address 0)",
                    pa
                )
            }
        } else {
            panic!("Invalid physical address {} for pagetable", pa);
        }
    }
}

/// Adds two signed values, returning an unsigned value or `None` if it would overflow.
/// 将两个有符号整数相加，并在结果可安全表示为无符号整数时返回该值；
/// 如果发生溢出，或者结果为负导致无法转换为 `usize`，则返回 `None`。
fn checked_add_to_unsigned(a: isize, b: isize) -> Option<usize> {
    a.checked_add(b)?.try_into().ok()
}

/// Manages a level 1 page table using linear mapping, where every virtual address is either
/// unmapped or mapped to an IPA with a fixed offset.
/// 使用线性映射管理一个一级页表：每个虚拟地址要么未映射，要么映射到一个与之保持固定偏移的
/// IPA。
///
/// This assumes that the same linear mapping is used both for the page table being managed, and for
/// code that is managing it.
/// 这里有一个关键前提：
///
/// - 被管理页表本身所在的内存，必须服从同一套线性映射；
/// - 当前执行的页表管理代码，也必须通过同一套线性映射访问这些页表。
///
/// 否则，源码中“虚拟地址推导物理地址”以及“物理地址反推虚拟地址”的逻辑都将不成立。
#[derive(Debug)]
pub struct LinearMap<R: TranslationRegime> {
    mapping: Mapping<LinearTranslation<R::Attributes>, R>,
}

impl<R: TranslationRegime<Asid = (), VaRange = ()>> LinearMap<R> {
    /// Creates a new identity-mapping page table with the given root level and offset, for
    /// use in the given TTBR.
    /// 创建一个采用线性映射的页表，指定其根级别与偏移量，并用于给定的翻译体制。
    ///
    /// This will map any virtual address `va` which is added to the table to the physical address
    /// `va + offset`.
    /// 该页表中任何被加入映射的虚拟地址 `va`，最终都会映射到物理地址 `va + offset`。
    ///
    /// The `offset` must be a multiple of [`PAGE_SIZE`]; if not this will panic.
    /// `offset` 必须按页对齐，否则会 panic。
    pub fn new(rootlevel: usize, offset: isize, regime: R) -> Self {
        Self {
            mapping: Mapping::new(LinearTranslation::new(offset), rootlevel, regime),
        }
    }
}

impl<R: TranslationRegime<Asid = usize, VaRange = VaRange>> LinearMap<R> {
    /// Creates a new identity-mapping page table with the given ASID, root level and offset, for
    /// use in the given TTBR.
    /// 创建一个采用线性映射的页表，指定 ASID、根级别、偏移量以及 VA 范围。
    ///
    /// This will map any virtual address `va` which is added to the table to the physical address
    /// `va + offset`.
    /// 任意加入该页表的虚拟地址 `va` 都将映射到物理地址 `va + offset`。
    ///
    /// The `offset` must be a multiple of [`PAGE_SIZE`]; if not this will panic.
    /// `offset` 必须是 [`PAGE_SIZE`] 的整数倍，否则会 panic。
    pub fn with_asid(
        asid: usize,
        rootlevel: usize,
        offset: isize,
        regime: R,
        va_range: VaRange,
    ) -> Self {
        Self {
            mapping: Mapping::with_asid_and_va_range(
                LinearTranslation::new(offset),
                asid,
                rootlevel,
                regime,
                va_range,
            ),
        }
    }
}

impl<R: TranslationRegime> LinearMap<R> {
    /// Returns the size in bytes of the virtual address space which can be mapped in this page
    /// table.
    /// 返回当前页表可覆盖的虚拟地址空间大小（单位：字节）。
    ///
    /// This is a function of the chosen root level.
    /// 可覆盖范围取决于根页表级别 `rootlevel`。
    pub fn size(&self) -> usize {
        self.mapping.size()
    }

    /// Activates the page table by programming the physical address of the root page table into
    /// `TTBRn_ELx`, along with the provided ASID. The previous value of `TTBRn_ELx` is returned so
    /// that it may later be restored by passing it to [`deactivate`](Self::deactivate).
    /// 通过把根页表的物理地址以及对应 ASID 写入 `TTBRn_ELx` 来激活当前页表。
    /// 返回值是激活前 `TTBRn_ELx` 的旧值，后续可以传给 [`deactivate`](Self::deactivate)
    /// 进行恢复。
    ///
    /// In test builds or builds that do not target aarch64, the `TTBR0_EL1` access is omitted.
    /// 在测试构建或非 aarch64 目标平台上，不会真正访问 `TTBR0_EL1` 等系统寄存器。
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
    /// - 不会制造破坏 Rust 别名规则的重叠映射；
    /// - 当这些映射仍然需要时，页表对象本身不能被提前释放。
    pub unsafe fn activate(&mut self) -> usize {
        // SAFETY: We delegate the safety requirements to our caller.
        unsafe { self.mapping.activate() }
    }

    /// Deactivates the page table, by setting `TTBRn_ELx` to the provided value, and invalidating
    /// the TLB for this page table's configured ASID. The provided TTBR value should be the value
    /// returned by the preceding [`activate`](Self::activate) call.
    /// 使用提供的旧 `TTBRn_ELx` 值恢复先前页表，并为该页表对应 ASID 执行 TLB 失效操作，
    /// 从而停用当前页表。
    ///
    /// In test builds or builds that do not target aarch64, the `TTBR0_EL1` access is omitted.
    /// 在测试构建或非 aarch64 目标平台上，不会真正访问系统寄存器。
    ///
    /// # Safety
    ///
    /// The caller must ensure that the previous page table which this is switching back to doesn't
    /// unmap any memory which the program is using.
    /// 调用者必须保证：切回去的旧页表仍然覆盖程序继续执行所需的全部内存。
    pub unsafe fn deactivate(&mut self, previous_ttbr: usize) {
        // SAFETY: We delegate the safety requirements to our caller.
        unsafe {
            self.mapping.deactivate(previous_ttbr);
        }
    }

    /// Maps the given range of virtual addresses to the corresponding physical addresses with the
    /// given flags.
    /// 将给定虚拟地址范围按当前线性规则映射到对应物理地址范围，并写入给定属性标志。
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
    /// Returns [`MapError::InvalidVirtualAddress`] if adding the configured offset to any virtual
    /// address within the `range` would result in overflow.
    /// 如果 `range` 内任意虚拟地址在加上 `offset` 后发生溢出，返回
    /// [`MapError::InvalidVirtualAddress`]。
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    /// 如果地址区间起止颠倒，返回 [`MapError::RegionBackwards`]。
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    /// 如果地址超出当前页表根级别所能覆盖的虚拟地址范围，返回 [`MapError::AddressRange`]。
    ///
    /// Returns [`MapError::InvalidFlags`] if the `flags` argument has unsupported attributes set.
    /// 如果 `flags` 中含有当前映射方式不支持的属性，返回 [`MapError::InvalidFlags`]。
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation`] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    /// 如果目标范围与正在生效的映射重叠，且修改方式违反 break-before-make（BBM）规则，
    /// 返回 [`MapError::BreakBeforeMakeViolation`]。
    pub fn map_range(
        &mut self,
        range: &MemoryRegion,
        flags: R::Attributes,
    ) -> Result<(), MapError> {
        self.map_range_with_constraints(range, flags, Constraints::empty())
    }

    /// Maps the given range of virtual addresses to the corresponding physical addresses with the
    /// given flags, taking the given constraints into account.
    /// 在考虑额外约束条件 `constraints` 的前提下，把给定虚拟地址范围映射到对应物理地址范围。
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active. This function writes block and page entries, but only maps them if `flags`
    /// contains [`PagingAttributes::VALID`], otherwise the entries remain invalid.
    /// 调用时机和错误条件与 [`map_range`](Self::map_range) 相同；区别在于这里允许调用者显式
    /// 指定是否禁止 block 映射等附加约束。
    ///
    /// # Errors
    ///
    /// Returns [`MapError::InvalidVirtualAddress`] if adding the configured offset to any virtual
    /// address within the `range` would result in overflow.
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
        // 先把虚拟地址范围的起始地址通过线性规则转换为物理起始地址，
        // 然后交由底层通用 `Mapping` 按“起始 VA + 起始 PA + 长度”完成整段映射。
        let pa = self
            .mapping
            .translation()
            .virtual_to_physical(range.start())?;
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
    /// 如果给定范围没有正好对齐到当前已有映射边界，函数内部可能会先把较大的 block entry
    /// 拆成更细粒度的子表项，然后再调用更新函数。
    /// 如果只想遍历而不修改，也不希望触发拆分，请使用 [`walk_range`](Self::walk_range)。
    ///
    /// The updater function receives the following arguments:
    /// 更新函数会收到以下参数：
    ///
    /// - The virtual address range mapped by each page table descriptor. A new descriptor will
    ///   have been allocated before the invocation of the updater function if a page table split
    ///   was needed.
    /// - 当前描述符所覆盖的虚拟地址范围；若需要拆分页表，则在回调前已经完成分配。
    /// - A mutable reference to the page table descriptor that permits modifications.
    /// - 可修改的页表描述符引用。
    /// - The level of a translation table the descriptor belongs to.
    /// - 该描述符所在的翻译表层级。
    ///
    /// The updater function should return:
    /// 更新函数的返回值约定如下：
    ///
    /// - `Ok` to continue updating the remaining entries.
    /// - 返回 `Ok`：继续处理后续条目。
    /// - `Err` to signal an error and stop updating the remaining entries.
    /// - 返回 `Err`：立即停止，并向上传播错误。
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active.
    /// 与映射接口一样，任何可能触发 break-before-make 的修改通常都应在页表不活跃时进行。
    ///
    /// # Errors
    ///
    /// Returns [`MapError::PteUpdateFault`] if the updater function returns an error.
    /// 如果更新函数返回错误，则会包装成 [`MapError::PteUpdateFault`]。
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    /// 如果范围起止反向，则返回 [`MapError::RegionBackwards`]。
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    /// 如果范围超出页表支持的虚拟地址空间，则返回 [`MapError::AddressRange`]。
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation`] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    /// 如果修改会违反活跃映射上的 BBM 规则，则返回
    /// [`MapError::BreakBeforeMakeViolation`]。
    pub fn modify_range<F>(&mut self, range: &MemoryRegion, f: &F) -> Result<(), MapError>
    where
        F: Fn(&MemoryRegion, &mut UpdatableDescriptor<R::Attributes>) -> Result<(), ()> + ?Sized,
    {
        self.mapping.modify_range(range, f)
    }

    /// Applies the provided callback function to the page table descriptors covering a given
    /// memory range.
    /// 对覆盖给定内存范围的页表描述符执行只读回调遍历。
    ///
    /// The callback function receives the following arguments:
    /// 回调函数将收到以下参数：
    ///
    /// - The range covered by the current step in the walk. This is always a subrange of `range`
    ///   even when the descriptor covers a region that exceeds it.
    /// - 当前遍历步对应的地址范围。即便某个描述符覆盖范围大于目标区间，这里传入的也只会是
    ///   `range` 的一个子区间。
    /// - The page table descriptor itself.
    /// - 当前页表描述符。
    /// - The level of a translation table the descriptor belongs to.
    /// - 描述符所在层级。
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
    /// 如果回调返回错误，则封装为 [`MapError::PteUpdateFault`]。
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    /// 如果范围反向，则返回 [`MapError::RegionBackwards`]。
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    /// 如果范围超出页表可覆盖的虚拟地址空间，则返回 [`MapError::AddressRange`]。
    pub fn walk_range<F>(&self, range: &MemoryRegion, f: &mut F) -> Result<(), MapError>
    where
        F: FnMut(&MemoryRegion, &Descriptor<R::Attributes>, usize) -> Result<(), ()>,
    {
        self.mapping.walk_range(range, f)
    }

    /// Looks for subtables whose entries are all empty and replaces them with a single empty entry,
    /// freeing the subtable.
    /// 查找所有“内容全为空”的子页表，并把它们收缩回一个空条目，同时释放对应子表内存。
    ///
    /// This requires walking the whole hierarchy of pagetables, so you may not want to call it
    /// every time a region is unmapped. You could instead call it when the system is under memory
    /// pressure.
    /// 由于该操作需要遍历整棵页表层级树，成本相对较高，因此通常不建议每次解除映射后都立刻
    /// 调用；更适合在系统内存紧张、需要回收页表开销时统一进行。
    pub fn compact_subtables(&mut self) {
        self.mapping.compact_subtables();
    }

    /// Returns the physical address of the root table.
    /// 返回根页表的物理地址。
    ///
    /// This may be used to activate the page table by setting the appropriate TTBRn_ELx if you wish
    /// to do so yourself rather than by calling [`activate`](Self::activate). Make sure to call
    /// [`mark_active`](Self::mark_active) after doing so.
    /// 如果调用者不想使用 [`activate`](Self::activate)，而是希望自行设置 `TTBRn_ELx`，
    /// 那么可以先取出这里返回的物理地址，再手动装载寄存器；完成后应调用
    /// [`mark_active`](Self::mark_active) 告知库当前页表已处于活跃状态。
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
    /// 当调用者手动设置 TTBR 激活页表时，库本身并不知道这件事；调用此函数后，后续
    /// [`map_range`](Self::map_range) / [`modify_range`](Self::modify_range) 会按“页表正在使用中”
    /// 的语义执行额外检查，以避免违反 break-before-make 约束。
    ///
    /// It is called automatically by [`activate`](Self::activate).
    /// 若使用 [`activate`](Self::activate)，则会自动调用本函数。
    pub fn mark_active(&mut self) {
        self.mapping.mark_active();
    }

    /// Marks the page table as inactive.
    /// 将当前页表标记为“非活跃”。
    ///
    /// This may be called after manually disabling the use of the page table, such as by setting
    /// the relevant TTBR to a different address.
    /// 如果调用者自行切换了 TTBR，使当前页表不再被使用，那么应调用此函数同步状态。
    ///
    /// It is called automatically by [`deactivate`](Self::deactivate).
    /// 若使用 [`deactivate`](Self::deactivate)，则会自动调用本函数。
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
        MapError,
        paging::{BITS_PER_LEVEL, MemoryRegion, PAGE_SIZE},
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;
    const GIB_512_S: isize = 512 * 1024 * 1024 * 1024;
    const GIB_512: usize = 512 * 1024 * 1024 * 1024;
    const NORMAL_CACHEABLE: El1Attributes =
        El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE);

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        // 地址空间起始位置的 1 字节映射。
        let mut pagetable = LinearMap::with_asid(1, 1, 4096, El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        // 地址空间起始位置的两个页映射。
        let mut pagetable = LinearMap::with_asid(1, 1, 4096, El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        // 地址空间末尾位置的 1 字节映射。
        let mut pagetable = LinearMap::with_asid(1, 1, 4096, El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        // 映射整个有效地址空间。这里选用二级 block 大小整数倍的 offset，避免全部退化为页映射，
        // 否则测试会非常慢。
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable =
            LinearMap::with_asid(1, 1, LEVEL_2_BLOCK_SIZE as isize, El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );
    }

    #[test]
    fn map_valid_negative_offset() {
        // A single byte which maps to IPA 0.
        // 使用负偏移，使某个非零 VA 恰好映射到 IPA 0。
        let mut pagetable =
            LinearMap::with_asid(1, 1, -(PAGE_SIZE as isize), El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE + 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        // 两页连续虚拟地址映射到更低的物理地址范围。
        let mut pagetable =
            LinearMap::with_asid(1, 1, -(PAGE_SIZE as isize), El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        // 地址空间末尾的单字节映射，在负偏移下同样应成立。
        let mut pagetable =
            LinearMap::with_asid(1, 1, -(PAGE_SIZE as isize), El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        // 同样选用二级 block 对齐的负偏移，以验证大范围映射在负偏移情况下也能正常工作。
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable = LinearMap::with_asid(
            1,
            1,
            -(LEVEL_2_BLOCK_SIZE as isize),
            El1And0,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(LEVEL_2_BLOCK_SIZE, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut pagetable = LinearMap::with_asid(1, 1, 4096, El1And0, VaRange::Lower);

        // One byte, just past the edge of the valid range.
        assert_eq!(
            pagetable.map_range(
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
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );
    }

    #[test]
    fn map_invalid_offset() {
        let mut pagetable = LinearMap::with_asid(1, 1, -4096, El1And0, VaRange::Lower);

        // One byte, with an offset which would map it to a negative IPA.
        // 当 VA 太小且 offset 为负时，PA 计算结果会落到 0 以下，因此应报错。
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, 1), NORMAL_CACHEABLE),
            Err(MapError::InvalidVirtualAddress(VirtualAddress(0)))
        );
    }

    #[test]
    fn physical_address_in_range_ttbr0() {
        let translation = LinearTranslation::<El1Attributes>::new(4096);
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(8192)),
            NonNull::new(4096 as *mut PageTable<El1Attributes>).unwrap(),
        );
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(GIB_512 + 4096)),
            NonNull::new(GIB_512 as *mut PageTable<El1Attributes>).unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn physical_address_to_zero_ttbr0() {
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(4096));
    }

    #[test]
    #[should_panic]
    fn physical_address_out_of_range_ttbr0() {
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(-4096_isize as usize));
    }

    #[test]
    fn physical_address_in_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to one page above the bottom
        // of physical address space.
        // 测试 TTBR1 场景下，从物理地址反推高地址虚拟地址的逆映射是否正确。
        let translation = LinearTranslation::new(GIB_512_S + 4096);
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(8192)),
            NonNull::new((4096 - GIB_512_S) as *mut PageTable<El1Attributes>).unwrap(),
        );
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(GIB_512)),
            NonNull::new(-4096_isize as *mut PageTable<El1Attributes>).unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn physical_address_to_zero_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(GIB_512_S);
        translation.physical_to_virtual(PhysicalAddress(GIB_512));
    }

    #[test]
    #[should_panic]
    fn physical_address_out_of_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(GIB_512_S);
        translation.physical_to_virtual(PhysicalAddress(-4096_isize as usize));
    }

    #[test]
    fn virtual_address_out_of_range() {
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(-4096);
        let va = VirtualAddress(1024);
        assert_eq!(
            translation.virtual_to_physical(va),
            Err(MapError::InvalidVirtualAddress(va))
        )
    }

    #[test]
    fn virtual_address_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        // 测试高地址虚拟地址区间在正向变换时是否正确落到低物理地址区间。
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(GIB_512_S);

        // The first page in the region covered by TTBR1.
        // TTBR1 覆盖区间中的第一页应映射到物理地址 0。
        assert_eq!(
            translation.virtual_to_physical(VirtualAddress(0xffff_ff80_0000_0000)),
            Ok(PhysicalAddress(0))
        );
        // The last page in the region covered by TTBR1.
        // TTBR1 覆盖区间中的最后一页应映射到对应的末尾物理页。
        assert_eq!(
            translation.virtual_to_physical(VirtualAddress(0xffff_ffff_ffff_f000)),
            Ok(PhysicalAddress(0x7f_ffff_f000))
        );
    }

    #[test]
    fn block_mapping() {
        // Test that block mapping is used when the PA is appropriately aligned...
        // 当偏移使得目标 PA 满足更大粒度对齐时，应优先使用 block 映射。
        let mut pagetable = LinearMap::with_asid(1, 1, 1 << 30, El1And0, VaRange::Lower);
        pagetable
            .map_range(
                &MemoryRegion::new(0, 1 << 30),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        assert_eq!(
            pagetable.mapping.root.mapping_level(VirtualAddress(0)),
            Some(1)
        );

        // ...but not when it is not.
        // 若目标 PA 未满足 block 对齐条件，则应退回到更细粒度层级。
        let mut pagetable = LinearMap::with_asid(1, 1, 1 << 29, El1And0, VaRange::Lower);
        pagetable
            .map_range(
                &MemoryRegion::new(0, 1 << 30),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
            )
            .unwrap();
        assert_eq!(
            pagetable.mapping.root.mapping_level(VirtualAddress(0)),
            Some(2)
        );
    }

    fn make_map() -> LinearMap<El1And0> {
        let mut lmap = LinearMap::with_asid(1, 1, 4096, El1And0, VaRange::Lower);
        // Mapping VA range 0x0 - 0x2000 to PA range 0x1000 - 0x3000
        // 这里建立一个简单线性映射样例，便于后续测试修改接口。
        lmap.map_range(&MemoryRegion::new(0, PAGE_SIZE * 2), NORMAL_CACHEABLE)
            .unwrap();
        lmap
    }

    #[test]
    fn update_backwards_range() {
        let mut lmap = make_map();
        assert!(
            lmap.modify_range(&MemoryRegion::new(PAGE_SIZE * 2, 1), &|_range, entry| {
                entry.modify_flags(
                    El1Attributes::SWFLAG_0,
                    El1Attributes::from_bits(0usize).unwrap(),
                )
            },)
                .is_err()
        );
    }

    #[test]
    fn update_range() {
        let mut lmap = make_map();
        lmap.modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry| {
            if !entry.is_table() {
                entry.modify_flags(
                    El1Attributes::SWFLAG_0,
                    El1Attributes::from_bits(0usize).unwrap(),
                )?;
            }
            Ok(())
        })
        .unwrap();
        lmap.modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|range, entry| {
            if !entry.is_table() {
                assert!(entry.flags().contains(El1Attributes::SWFLAG_0));
                assert_eq!(range.end() - range.start(), PAGE_SIZE);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn breakup_invalid_block() {
        const BLOCK_RANGE: usize = 0x200000;

        let mut lmap = LinearMap::with_asid(1, 1, 0x1000, El1And0, VaRange::Lower);
        // 先建立一个无效 block，并带一个软件标志；随后把其中第一页单独映射为有效页，
        // 用于验证 block 拆分后各页属性是否正确继承/覆盖。
        lmap.map_range(
            &MemoryRegion::new(0, BLOCK_RANGE),
            NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::SWFLAG_0,
        )
        .unwrap();
        lmap.map_range(
            &MemoryRegion::new(0, PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
        )
        .unwrap();
        lmap.modify_range(&MemoryRegion::new(0, BLOCK_RANGE), &|range, entry| {
            if entry.level() == 3 {
                let has_swflag = entry.flags().contains(El1Attributes::SWFLAG_0);
                let is_first_page = range.start().0 == 0usize;
                assert!(has_swflag != is_first_page);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn split_live_block_mapping() -> () {
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut lmap = LinearMap::with_asid(1, 1, BLOCK_SIZE as isize, El1And0, VaRange::Lower);
        // 该测试验证：当 block 映射已经处于 live 状态时，若后续修改需要拆分该 block，
        // 会触发 break-before-make 相关的异常路径。
        lmap.map_range(
            &MemoryRegion::new(0, BLOCK_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::READ_ONLY
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
        )
        .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { lmap.activate() };
        lmap.map_range(
            &MemoryRegion::new(0, PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::READ_ONLY
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
        )
        .unwrap();
        lmap.map_range(
            &MemoryRegion::new(PAGE_SIZE, 2 * PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::READ_ONLY
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
        )
        .unwrap();
        let r = lmap.map_range(
            &MemoryRegion::new(PAGE_SIZE, 2 * PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
        );
        unsafe { lmap.deactivate(ttbr) };
        r.unwrap();
    }
}
