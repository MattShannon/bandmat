if ($env:APPVEYOR_REPO_TAG -ne "true")
{
    $version = "dev-$($env:APPVEYOR_REPO_COMMIT.substring(0,8))";
}
else
{
    $version = "$($env:APPVEYOR_REPO_TAG_NAME)"
}
Update-AppveyorBuild -Version "$version"
