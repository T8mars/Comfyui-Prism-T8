"""Optional allocator operations across PyTorch builds; imports remain torch-free."""


def host_cache_release_supported(torch_module):
    return callable(getattr(getattr(torch_module, '_C', None), '_host_emptyCache', None))


def empty_host_cache(torch_module):
    # Some CPU/Windows wheels do not export this private CUDA allocator hook.
    # Missing support is not evidence that host pages were returned to the OS.
    empty = getattr(getattr(torch_module, '_C', None), '_host_emptyCache', None)
    if not callable(empty):
        return False
    empty()
    return True
