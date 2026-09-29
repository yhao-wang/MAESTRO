import importlib.abc
import importlib.machinery
import os
import sys

# Putting maestro/patches/vllm on PYTHONPATH and setting MAESTRO_RUNTIME_VLLM_PATCH=1
# patches vLLM at interpreter startup. Set MAESTRO_RUNTIME_VLLM_PATCH_DEFER=1 to let
# the rollout server apply speculative-decoding patches after Ray actors are
# alive. Under defer, keep generic Ray worker startup light: do not import
# speculative_decode/torch/vLLM here.  Instead install a small import hook that
# applies the PaddleJob vLLM port patch only once a vLLM networking/executor
# module is actually imported.
_VLLM_PORT_PATCH_TARGETS = {
    "vllm.utils.network_utils",
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.executor.ray_executor",
    "vllm.v1.executor.ray_executor_v2",
    "vllm.v1.executor.uniproc_executor",
    "vllm.v1.utils",
}
_APPLYING_PORT_PATCH = False
_IMPORT_HOOK_INSTALLED = False


def _apply_deferred_port_patch() -> None:
    global _APPLYING_PORT_PATCH
    if _APPLYING_PORT_PATCH:
        return
    _APPLYING_PORT_PATCH = True
    try:
        from speculative_decode import apply_port_patch

        apply_port_patch()
    except Exception as exc:
        print(f"[opd-rollout] failed to apply deferred vLLM port patch: {exc}", file=sys.stderr)
    finally:
        _APPLYING_PORT_PATCH = False


class _MaestroVllmPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module) -> None:
        self._wrapped_loader.exec_module(module)
        _apply_deferred_port_patch()


class _MaestroVllmPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _VLLM_PORT_PATCH_TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if not isinstance(spec.loader, _MaestroVllmPatchLoader):
            spec.loader = _MaestroVllmPatchLoader(spec.loader)
        return spec


def _install_deferred_port_patch_hook() -> None:
    global _IMPORT_HOOK_INSTALLED
    if _IMPORT_HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _MaestroVllmPatchFinder())
    _IMPORT_HOOK_INSTALLED = True


if os.environ.get("MAESTRO_RUNTIME_VLLM_PATCH") == "1":
    if os.environ.get("MAESTRO_RUNTIME_VLLM_PATCH_DEFER") == "1":
        _install_deferred_port_patch_hook()
    else:
        from speculative_decode import apply_patches

        apply_patches()
