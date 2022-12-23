# Loop through the user list
Get-Content "BINXJIA.txt" | %{

  # Define what each job does
  $ScriptBlock = {
    param($pipelinePassIn) 
    .\bin\Release\ModuleUsageStat.exe user $pipelinePassIn 30 "$pipelinePassIn.stat"
    Start-Sleep 10
  }

  # Execute the jobs in parallel
  Start-Job -Init ([ScriptBlock]::Create("Set-Location '$pwd'")) -ScriptBlock $ScriptBlock -ArgumentList $_
}

Get-Job

# Wait for it all to complete
While (Get-Job -State "Running")
{
  Start-Sleep 10
}

# Getting the information back from the jobs
Get-Job | Receive-Job