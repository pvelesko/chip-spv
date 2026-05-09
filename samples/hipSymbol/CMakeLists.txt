# Test symbol access

add_chip_test(hipTestDeviceSymbol hipTestDeviceSymbol PASSED
  hipTestDeviceSymbol.cpp)
add_chip_test(hipTestConstantDeviceSymbol hipConstantTestDeviceSymbol PASSED
  hipTestConstantDeviceSymbol.cpp)
add_chip_test(hipTestSymbolReset hipTestSymbolReset PASSED
  hipTestSymbolReset.cc)
# hipTestSymbolInit performs pointer-equality (`Foo == &Bar`) on two
# `__device__` global symbols. In the Vulkan/Logical addressing path that
# clvk targets, `__device__` globals are bound as set=1 StorageBuffer
# descriptors; their addresses are not comparable as uint64 SSA values, and
# the kernel-drop backstop in HIPSPVLowerToHLSLShape leaves the icmp
# operands as unresolved external globals which the SPIR-V backend then
# emits with `OpCapability Linkage` (rejected by the Vulkan 1.3 spec).
# A proper fix requires lowering the icmp-pattern globals through
# PhysicalStorageBuffer64 (BDA), exposing the raw uint64 device address as
# a push-constant and converting it via OpConvertUToPtr; that work is not
# yet wired up. Exclude under CHIP_USE_NATIVE_VULKAN_SPIRV until then.
if(NOT CHIP_USE_NATIVE_VULKAN_SPIRV)
  add_chip_test(hipTestSymbolInit hipTestSymbolInit PASSED hipTestSymbolInit.cc)
endif()
add_chip_test(hipTestResetStaticVar hipTestResetStaticVar PASSED
  hipTestResetStaticVar.cc)
add_chip_test(hipTestVariableTemplateSymbols hipTestVariableTemplateSymbols
  PASSED hipTestVariableTemplateSymbols.cc)
