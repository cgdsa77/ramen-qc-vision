Param(
    [string]$InputDir = "data/raw/抻面",
    [string]$OutputRoot = "data/processed/抻面",
    [double]$Fps = 2
)

# 确保以仓库根目录为当前路径，避免相对路径失效
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

if (-not (Test-Path $InputDir)) {
    Write-Error "输入目录不存在: $InputDir"
    exit 1
}

Get-ChildItem $InputDir -Filter *.mp4 | ForEach-Object {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $outDir = Join-Path $OutputRoot $name
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $outPattern = Join-Path $outDir ($name + "_%05d.jpg")
    Write-Host "抽帧 -> $($outDir)"
    ffmpeg -i $_.FullName -vf "fps=$Fps" $outPattern
}

Write-Host "完成。输出路径: $OutputRoot"

