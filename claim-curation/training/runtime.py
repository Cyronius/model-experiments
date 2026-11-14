"""Pinned-runtime settings shared by training and paired generation evaluation."""


def configure(torch):
    settings = {"hip": torch.version.hip, "native_triton_overrides_disabled": False}
    if torch.version.hip and getattr(torch, "_native", None) is not None:
        # The isolated AMD 2.13 wheel enables optional Python-native Triton
        # replacements for ordinary ATen ops. They need a host C development
        # sysroot, absent here. Use the wheel's precompiled HIP/ATen operators
        # in both base and adapter runs; no model math or weights are changed.
        from torch._native.triton_utils import deregister_op_overrides
        deregister_op_overrides()
        settings["native_triton_overrides_disabled"] = True
    return settings
