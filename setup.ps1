$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host 'Creating the DAvtar virtual environment...'
python -m venv venv
$python = Join-Path $root 'venv\Scripts\python.exe'

Write-Host 'Installing root Python dependencies...'
& $python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip tooling' }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Failed to install root requirements' }
& $python -m pip install gdown
if ($LASTEXITCODE -ne 0) { throw 'Failed to install gdown' }

Write-Host 'Downloading core checkpoints...'
& $python download_models.py
if ($LASTEXITCODE -ne 0) { throw 'Core checkpoint download failed' }

Write-Host 'Installing Real-ESRGAN and its dependencies...'
Push-Location (Join-Path $root 'Real-ESRGAN')
try {
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install Real-ESRGAN requirements' }
    & $python setup.py develop
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install Real-ESRGAN' }
}
finally {
    Pop-Location
}

# Real-ESRGAN downloads model files lazily during inference. Run it once
# against an empty input directory so setup completes the downloads without
# processing a video or image.
$probeInput = Join-Path $root 'temp\setup_weight_probe_input'
$probeOutput = Join-Path $root 'temp\setup_weight_probe_output'
New-Item -ItemType Directory -Force -Path $probeInput, $probeOutput | Out-Null
Push-Location (Join-Path $root 'Real-ESRGAN')
try {
    Write-Host 'Downloading Real-ESRGAN, GFPGAN, and face-helper weights...'
    & $python inference_realesrgan.py `
        -n RealESRGAN_x4plus `
        -i $probeInput `
        --output $probeOutput `
        --outscale 3.5 `
        --face_enhance
    if ($LASTEXITCODE -ne 0) { throw 'HD model-weight download failed' }
}
finally {
    Pop-Location
}

Remove-Item -LiteralPath $probeInput, $probeOutput -Recurse -Force -ErrorAction SilentlyContinue

$requiredFiles = @(
    'checkpoints\wav2lip_gan.pth',
    'checkpoints\face_segmentation.pth',
    'checkpoints\esrgan_yunying.pth',
    'checkpoints\pretrained.state',
    'Real-ESRGAN\weights\RealESRGAN_x4plus.pth',
    'Real-ESRGAN\gfpgan\weights\detection_Resnet50_Final.pth',
    'Real-ESRGAN\gfpgan\weights\parsing_parsenet.pth',
    'venv\Lib\site-packages\gfpgan\weights\GFPGANv1.3.pth'
)
foreach ($relativePath in $requiredFiles) {
    $absolutePath = Join-Path $root $relativePath
    if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
        throw "Required model file is missing after setup: $relativePath"
    }
}

Write-Host 'DAvtar setup completed. All core and HD model weights are installed.'
