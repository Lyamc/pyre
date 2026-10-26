# One-shot mechanical port of backend-generic Burn code (<= 0.21: Tensor<B, D>, Module<B>)
# to the device-dispatch API (0.22+/Lyamc fork: Tensor<D>, Module, Device). Manual fixes follow.
param([string[]]$Dirs)
foreach ($d in $Dirs) {
  Get-ChildItem $d -Recurse -Filter *.rs | ForEach-Object {
    $s = [IO.File]::ReadAllText($_.FullName); $o = $s
    $s = $s -creplace 'use burn::tensor::backend::Backend;\r?\n', ''
    $s = $s -creplace 'use burn::prelude::Backend;\r?\n', ''
    $s = $s -creplace '\bbackend::Backend,\s*', ''
    $s = $s -creplace ',\s*backend::Backend\b', ''
    $s = $s -creplace '\bB::Device\b', 'burn::tensor::Device'
    $s = $s -creplace '<B: Backend(\s*\+\s*[\w:]+)*>', ''
    $s = $s -creplace '<B: Backend(\s*\+\s*[\w:]+)*,\s*', '<'
    $s = $s -creplace ',\s*B: Backend(\s*\+\s*[\w:]+)*>', '>'
    $s = $s -creplace ',\s*B: Backend(\s*\+\s*[\w:]+)*,', ','
    $s = $s -creplace '::<B>', ''
    $s = $s -creplace '::<B,\s*', '::<'
    $s = $s -creplace '<B>', ''
    $s = $s -creplace '<B,\s*', '<'
    $s = $s -creplace ',\s*B>', '>'
    $s = $s -creplace '(?m)^\s*B: Backend[^\r\n]*,\r?\n', ''
    $s = $s -creplace '\bB::seed\(([^,]+),\s*', '${1}.seed('
    if ($s -ne $o) { [IO.File]::WriteAllText($_.FullName, $s) }
  }
}
