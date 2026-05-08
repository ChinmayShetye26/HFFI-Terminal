$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\frontend")
npm.cmd run dev
