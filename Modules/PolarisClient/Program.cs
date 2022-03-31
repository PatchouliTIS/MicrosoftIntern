using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Mono.Options;
using LegoPortal.Data.Polaris;

namespace PolarisClient
{
    class Program
    {
        private const string DefaultJobName = "DefaultJob";
        private const string DefaultUser = "Anonymous";

        static void ShowHelp(OptionSet p)
        {
            Console.WriteLine("Usage: Submit an experiment and wait for the result");
            Console.WriteLine();
            Console.WriteLine("Options:");
            p.WriteOptionDescriptions(Console.Out);
        }

        private static void AddOptional(OptionSet p, string prototype, string description, string defaultValue,
            bool required, Action<string> a)
        {
            StringBuilder builder = new StringBuilder();
            if (required)
            {
                builder.Append("<Required> ");
            }
            else
            {
                builder.Append("<Optional> ");
            }
            builder.Append(description);

            if (defaultValue != null)
            {
                builder.Append(" ");
                builder.Append($"Default vaulue is \"{defaultValue}\"");
            }
            p.Add(prototype,
                builder.ToString(),
                a);
        }

        static int Main(string[] args)
        {
            var host = System.Configuration.ConfigurationManager.AppSettings["apiHost"];
            var caseConfig = System.Configuration.ConfigurationManager.AppSettings["caseConfig"];
            var caseConfigGPU = "https://cosmos09.osdinfra.net/cosmos/IndexServePlatform/local/Polaris/PolarisBox/CaseConfigFiles/InferenceService/InferenceServiceV15WinGPU.ini";
            var totalRunningTimeInHour = int.Parse(System.Configuration.ConfigurationManager.AppSettings["totalRunningWaitTimeInHour"]);
            var maxResubmitTimes = int.Parse(System.Configuration.ConfigurationManager.AppSettings["maxResubmitTimes"]);
            if (string.IsNullOrEmpty(host))
            {
                throw new ApplicationException("the apiHost Setting in configuration is missed or empty");
            }
            var condaEnv = "https://cosmos09.osdinfra.net/cosmos/DLISModelRepository/local/Models/yuazhan/conda/ortgpu_1.4_tokenizers_0.7.0_vcomp.zip";
            var envVar = "KMP_AFFINITY=respect,none;PYTHONUNBUFFERED=1;";
            var outFile = string.Empty;
            var outQueryOutputLog = string.Empty;
            var outPolarisLink = string.Empty;
            string passFailPath = string.Empty;
            bool alwaysPass = false;

            var modelPath = string.Empty;
            var testQuery = string.Empty;
            var maxCpuCores = "2";
            var totalCpuCores = "6";
            var serviceType = "InferenceServiceV15WinError";

            var user = Program.DefaultUser;
            var name = Program.DefaultJobName;
            var note = "none";
            var machineCount = 1;
            var priority = 2;

            var cmd = "run.cmd http";
            var modelName = "UntitledModel";
            var showHelp = false;

            var hasTraceId = "false";
            var QPSList = "-1";
            var timeoutInMs = "100";
            var waitingModelReadyInMin = "10";

            var maxDataSizeInBytes = "65536";
            var maxMemoryUsageInMB = "8000";
            var maxBatchSize = "1";

            var modelServingClientType = "HttpBased";
            var sharedMemoryClientType = "Standard";
            var isThreadSafe = "true";
            var parallelism = "1";
            var queryClientParallelism = "1";

            var p = new OptionSet()
            {
                {
                    "mp|modelPath=",
                    "<Required> The {CosmosUrlToModel}",
                    v => modelPath = string.IsNullOrEmpty(v) ? modelPath : v
                },
                {
                    "qp|queryPath=",
                    "<Required> The {CosmosUrlToTestQueries}",
                    v => testQuery = string.IsNullOrEmpty(v) ? testQuery : v
                },
                {
                    "ce|condaEnvPath=",
                    $"<Optional> The {{condaEnvPath}}. Default value is \"{condaEnv}\".",
                    v => condaEnv = string.IsNullOrEmpty(v) ? condaEnv : v
                },
                                {
                    "ev|EnvVariable=",
                    $"<Optional> The {{EnvVariable}}. Default value is \"{envVar}\".",
                    v => envVar = string.IsNullOrEmpty(v) ? envVar : v
                },
                {
                    "st|serviceType=",
                    $"<Optional> The {{serviceType}}. Default value is \"{serviceType}\".",
                    v => serviceType = string.IsNullOrEmpty(v) ? serviceType : (v=="V100"?"DLIS-Win-GPU-V100":(v=="CPU"?"InferenceServiceWin":(v=="T4"?"DLIS-Win-GPU-T4":(v=="A100"?"DLIS-Win-GPU-A100":(v=="M60"?"DLIS-Win-GPU-M60" : serviceType)))))

                },
                {
                    "o|out=",
                    "<Required> The {OutputPath} of response json content",
                    v => outFile = string.IsNullOrEmpty(v) ? outFile :v
                },
                {
                    "oq|outQueryOutputLog=",
                    "<Required> the {OutputPath} for Query Outputs Logs",
                    v => outQueryOutputLog = string.IsNullOrEmpty(v) ? outQueryOutputLog : v
                },
                {
                    "op|outPolarisLink=",
                    "<Required> the {OutputPath} for a link to full Polaris test results.",
                    v => outPolarisLink = string.IsNullOrEmpty(v) ? outPolarisLink : v
                },
                 {
                    "ap|AlwaysPass",
                    "<Required> Set the module to always pass.  Pass/fail will then be reported through outPassFail (which must also be set).  This setting allows debug logs to be reported, since a module failure prevents Aether from recording any module outputs.",
                    v => alwaysPass = (v != null)
                },
                {
                    "pf|outPassFail=",
                    "<Required> the {OutputPath} to a file indicating whether this module passes or fails.",
                    v => passFailPath = string.IsNullOrEmpty(v) ? passFailPath : v
                },
                {
                    "u|user=",
                    "<Optional> The {UserName} of request json content. Default value is \"{user}\"",
                    (v) =>
                    {
                        user = string.IsNullOrEmpty(v) ? user : v;
                        // remove the domain
                        if (user.Contains("\\"))
                        {
                            user = user.Substring(user.LastIndexOf('\\') + 1);
                        }
                    }
                },
                {
                    "name|n=",
                    $"<Optional> The {{JobName}}. Default value is \"{name}\".",
                    v => name = string.IsNullOrEmpty(v) ? name :v
                },
                {
                    "note=",
                    $"<Optional> The {{Description}} of the job. Default value is \"{note}\".",
                    v => note = string.IsNullOrEmpty(v) ? note:v
                },
                {
                    "totalCpuCores=",
                    $"<Optional> The {{TotalCpuCores}} number. Default value is \"{totalCpuCores}\".",
                    v => totalCpuCores = string.IsNullOrEmpty(v) ? totalCpuCores : v
                },
                {
                    "maxCpuCores=",
                    $"<Optional> The {{MaxCpuCores}} number. Default value is \"{maxCpuCores}\".",
                    v => maxCpuCores = string.IsNullOrEmpty(v) ? maxCpuCores : v
                },
                {
                    "priority=",
                    $"<Optional> The {{priority}} number. Default value is \"{priority}\".",
                    v => priority = string.IsNullOrEmpty(v) ? priority : int.Parse(v)
                },
                {
                    "modelname=",
                    $"<Optional> The {{modelName}}. Default Value is \"{modelName}\".",
                    v => modelName = string.IsNullOrEmpty(v) ? modelName : v
                },
                {
                    "cmd=",
                    $"<Optional> The {{CommandLine}} to start the process. Default value is \"{cmd}\".",
                    v => cmd = string.IsNullOrEmpty(v) ? cmd :v
                },
                {
                    "hasTraceId=",
                    $"<Optional> The {{hasTraceId}}. Default value is \"{hasTraceId}\".",
                    v => hasTraceId = string.IsNullOrEmpty(v) ? hasTraceId :v
                },                
                {
                    "qpsList=",
                    $"<Optional> The {{QPSList}}. Default value is \"{QPSList}\".",
                    v => QPSList = string.IsNullOrEmpty(v) ? QPSList :v
                },
                {
                    "timeoutInMs=",
                    $"<Optional> The {{TimeoutInMs}}. Default value is \"{timeoutInMs}\".",
                    v => timeoutInMs = string.IsNullOrEmpty(v) ? timeoutInMs :v
                },
                {
                    "waitingModelReadyInMs=",
                    $"<Optional> The {{WaitingModelReadyInMs}}. Default value is \"{waitingModelReadyInMin}\".",
                    v => waitingModelReadyInMin = string.IsNullOrEmpty(v) ? waitingModelReadyInMin :v
                },
                {
                    "maxMemoryUsageInMB=",
                    $"<Optional> The {{MaxMemoryUsageInMB}}. Default value is \"{maxMemoryUsageInMB}\".",
                    v => maxMemoryUsageInMB = string.IsNullOrEmpty(v) ? maxMemoryUsageInMB :v
                },
                {
                    "maxDataSizeInBytes=",
                    $"<Optional> The {{MaxDataSizeInBytes}}. Default value is \"{maxDataSizeInBytes}\".",
                    v => maxDataSizeInBytes = string.IsNullOrEmpty(v) ? maxDataSizeInBytes :v
                },
                {
                    "maxBatchSize=",
                    $"<Optional> The {{MaxBatchSize}}. Default value is \"{maxBatchSize}\".",
                    v => maxBatchSize = string.IsNullOrEmpty(v) ? maxBatchSize :v
                },
                {
                    "modelServingClientType=",
                    $"<Optional> The {{ModelServingClientType}}. Default value is \"{modelServingClientType}\".",
                    v => modelServingClientType = string.IsNullOrEmpty(v) ? modelServingClientType :v
                },
                {
                    "sharedMemoryClientType=",
                    $"<Optional> The {{SharedMemoryClientType}}. Default value is \"{sharedMemoryClientType}\".",
                    v => sharedMemoryClientType = string.IsNullOrEmpty(v) ? sharedMemoryClientType :v
                },
                {
                    "isThreadSafe=",
                    $"<Optional> The {{IsThreadSafe}}. Default value is \"{isThreadSafe}\".",
                    v => isThreadSafe = string.IsNullOrEmpty(v) ? isThreadSafe :v
                },
                {
                    "parallelism=",
                    $"<Optional> The {{Parallelism}}. Default value is \"{parallelism}\".",
                    v => parallelism = string.IsNullOrEmpty(v) ? parallelism :v
                },
                {
                    "queryCleintParallelism=",
                    $"<Optional> The {{QueryCleintParallelism}}. Default value is \"{queryClientParallelism}\".",
                    v => queryClientParallelism = string.IsNullOrEmpty(v) ? queryClientParallelism :v
                },
                {
                    "h|help",
                    "show this message and exit",
                   v => showHelp = v != null
                },

            };

            var additionalParameter = p.Parse(args);

            // Aether's default command line stuff is tacked on with a different syntax.
            string[] aetherParameters = new string[] { "ExperimentID", "Owner", "NodeId" };
            string[] aetherParameterValues = Program.ExtractAetherParameters(args, aetherParameters);

            string experimentId = aetherParameterValues[0];
            string owner = aetherParameterValues[1];
            string nodeId = aetherParameterValues[2];

            // If we get parameter values from Aether but not on the command line, use the Aether-derived ones instead:
            if ((!string.IsNullOrEmpty(owner)) && user == Program.DefaultUser)
            {
                user = owner;
            }

            if ((!string.IsNullOrEmpty(experimentId) && name == Program.DefaultJobName))
            {
                name = string.Format($"Aether {experimentId}");
            }

            // For logging clarity, strip out Aether parameters we know from the "Additional Parameter" list.
            additionalParameter = additionalParameter.Where(a =>
            {
                foreach (string aetherParam in aetherParameters)
                {
                    if (a.ToLower().StartsWith(aetherParam.ToLower()))
                    {
                        return false;
                    }
                }
                return true;
            }).ToList();

            Console.WriteLine("Unknown parameters: {0}", string.Join(" ", additionalParameter));

            if (showHelp || args.Length == 0)
            {
                ShowHelp(p);
                return -1;
            }

            if (string.IsNullOrEmpty(modelPath))
            {
                Console.Error.WriteLine("modelPath can't be null or empty");
                ShowHelp(p);
                return -1;
            }

            if (string.IsNullOrEmpty(testQuery))
            {
                Console.Error.WriteLine("queryPath can't be null or empty");
                ShowHelp(p);
                return -1;
            }

            // Allow Aether to call this program directly by reading input parameters from local files, if inputs were specified in this form.
            modelPath = Program.SubstituteLocalFileContentsIfNeeded(modelPath);
            testQuery = Program.SubstituteLocalFileContentsIfNeeded(testQuery);

            if (string.IsNullOrEmpty(outFile))
            {
                Console.Error.WriteLine("out can't be null or empty");
                ShowHelp(p);
                return -1;
            }

            if (string.IsNullOrEmpty(outQueryOutputLog))
            {
                Console.Error.WriteLine("Gentle reminder: outQueryOutputLog is not set.  You will have to dig through the Polaris web UI to discover it.");
            }

            if (string.IsNullOrEmpty(outPolarisLink))
            {
                Console.Error.WriteLine("Gentle reminder: outPolarisLink is not set.  The link can be found in stdout text.");
            }

            if (alwaysPass && string.IsNullOrEmpty(passFailPath))
            {
                Console.Error.WriteLine("If you set AlwaysPass, you must also set an outPassFail path so the larger experiment can detect and handle failures.");
                ShowHelp(p);
                return -1;
            }

            if (cmd == "run.cmd")
            {
                cmd = "run.cmd http";
            }

            cmd = cmd.Replace("#SEP#", " ");

            if (condaEnv.StartsWith("https") == false)
            {
                condaEnv = "";
            }

            if(serviceType == "InferenceServiceV15WinError")
            {
                throw new ApplicationException("Unsupported Machine Type");
            }

            TimeSpan waitTime = new TimeSpan(0, totalRunningTimeInHour, 0, 0);
            PolarisV2Proxy.Init(host);
            var job = new JobNewDTO();

            if(serviceType != "InferenceServiceWin")
            {
                job.Type = "InferenceServiceWinGpu";
                job.MachineType = serviceType;
            }
            else
            {
                job.Type = serviceType;
                job.MachineType = "DLIS-Win-V15";
            }

            job.Name = name;
            job.User = user;
            job.Note = note;
            job.MachineCount = machineCount;
            job.Priority = priority;
            job.Origin = "PolarisClientAPI";
            job.Tests = new List<JobTestNewDTO>();
            var jobTest = new JobTestNewDTO();
            jobTest.Name = $"{name}_1";
            jobTest.Settings = new Dictionary<string, string>();
            //jobTest.Settings.Add("CaseConfig", caseConfig);
            jobTest.Settings.Add("Cmd", cmd);
            jobTest.Settings.Add("CondaEnvironment", condaEnv);
            jobTest.Settings.Add("EnvironmentVariables", envVar);
            jobTest.Settings.Add("InstancesNum", "1");
            jobTest.Settings.Add("MaxCpuCores", maxCpuCores);
            jobTest.Settings.Add("MaxDataSizeInBytes", maxDataSizeInBytes);
            jobTest.Settings.Add("MaxMemoryUsageInMB", maxMemoryUsageInMB);
            jobTest.Settings.Add("ModelName", modelName.ToLower());
            jobTest.Settings.Add("ModelPath", modelPath);
            jobTest.Settings.Add("ModelTypeName", "DummyWindowsLibrary.DummyWindowsLibrary");
            jobTest.Settings.Add("ModelVersion", "1");
            jobTest.Settings.Add("PerfFlag", "APCounter");
            jobTest.Settings.Add("QPSList", QPSList);
            jobTest.Settings.Add("HasTraceId", hasTraceId);
            jobTest.Settings.Add("TestQuery", testQuery);
            jobTest.Settings.Add("TimeoutInMs", timeoutInMs);
            jobTest.Settings.Add("TotalCpuCores", totalCpuCores);
            jobTest.Settings.Add("WaitingModelReadyInMin", waitingModelReadyInMin);
            jobTest.Settings.Add("MaxBatchSize", maxBatchSize);
            jobTest.Settings.Add("ModelServingClientType", modelServingClientType);
            if(modelServingClientType == "SharedMemoryBased")
            {
                jobTest.Settings.Add("SharedMemoryClientType", sharedMemoryClientType);
            }
            jobTest.Settings.Add("IsThreadSafe", isThreadSafe);
            jobTest.Settings.Add("Parallelism", parallelism);
            jobTest.Settings.Add("QueryCleintParallelism", queryClientParallelism);
            job.Tests.Add(jobTest);

            var requestString = JsonConvert.SerializeObject(job, Formatting.Indented);
            Console.WriteLine("[{0}] Submit the following request", DateTime.Now);
            Console.WriteLine(requestString);
            int currentSubmitTime = 0;
            DateTime startTime = DateTime.Now;
            while (currentSubmitTime < maxResubmitTimes)
            {
                var jobId = PolarisV2Proxy.SubmitJobUsingDTO(requestString);
                var polarisUrl = string.Concat("http://lego/polaris/jobs/", jobId);
                Console.WriteLine("[{0}] Submitted successfully. The Job Id is {1}", DateTime.Now, jobId);
                Console.WriteLine("[{0}] Tracking Url is {1}", DateTime.Now, polarisUrl);
                if (!string.IsNullOrEmpty(outPolarisLink))
                {
                    File.WriteAllText(outPolarisLink, polarisUrl + Environment.NewLine);
                }

                bool successed = false;
                while (true)
                {
                    var responseText = PolarisV2Proxy.FetchJobResult(jobId);
                    var statusObject = (JObject)JsonConvert.DeserializeObject(responseText);
                    var statusText = statusObject.GetValue("Status").ToString();
                    if (statusText == "Finished")
                    {
                        var errorMessage = statusObject.GetValue("Error").ToString() ?? string.Empty;
                        errorMessage = errorMessage.Trim();
                        if (string.IsNullOrEmpty(errorMessage))
                        {
                            string json = JsonConvert.SerializeObject(statusObject, Formatting.Indented);
                            File.WriteAllText(outFile, json);
                            Console.WriteLine("Job Finished");
                            successed = true;
                        }
                        else
                        {
                            string json = JsonConvert.SerializeObject(statusObject, Formatting.Indented);
                            File.WriteAllText(outFile, json);
                            Console.Error.WriteLine("Job finished with error. Error message: {0}", errorMessage);
                        }

                        // Copy the Docker log to an accessible place regardless whether this job succeeded or failed.
                        string detailedLogs = PolarisV2Proxy.GetTestLogsLocation(statusObject);
                        if (!(string.IsNullOrEmpty(detailedLogs) || string.IsNullOrEmpty(outQueryOutputLog)))
                        {
                            Program.CopyDockerLogToOutput(outQueryOutputLog, detailedLogs);
                        }
                        break;
                    }
                    if (statusText != "OnGoing" && statusText != "Ready")
                    {
                        Console.Error.WriteLine("Something wrong with this job");
                        return Program.RecordPassFail(-1, alwaysPass, passFailPath);
                    }

                    var time = DateTime.Now - startTime;
                    if (time > waitTime)
                    {
                        Console.Error.WriteLine($"Time out. Job {jobId} runs too long");
                        return Program.RecordPassFail(-1, alwaysPass, passFailPath);
                    }
                    Console.WriteLine("[{0}] Job is running", DateTime.Now);
                    // Check every 5 mins
                    Thread.Sleep(1000 * 300);
                }
                if (successed)
                {
                    return Program.RecordPassFail(0, alwaysPass, passFailPath);
                }
                currentSubmitTime++;
                Console.WriteLine("Retry to submit");
            }
            Console.Error.WriteLine("Job fail after retry {0} times", maxResubmitTimes);
            return Program.RecordPassFail(-1, alwaysPass, passFailPath);
        }

        /// <summary>
        /// Copies any docker log created to the output file requested from the command line.
        /// </summary>
        /// <param name="localOutputPath"></param>
        /// <param name=""></param>
        private static void CopyDockerLogToOutput(string localOutputPath, string cosmosLogPath)
        {

            // Extract the VCName from Cosmos Path
            const string cosmosPart = "/cosmos/";
            int endOfCosmosIndex = cosmosLogPath.IndexOf(cosmosPart, StringComparison.OrdinalIgnoreCase) + cosmosPart.Length;
            int vcLength = cosmosLogPath.IndexOf('/', endOfCosmosIndex);
            string vcName = cosmosLogPath.Substring(0, vcLength + 1);

            string vcBit = vcName.Replace("https:", "vc:").Replace("/cosmos/", "/").Replace(".osdinfra.net", string.Empty);
            while (vcBit.EndsWith("/"))
            {
                vcBit = vcBit.Substring(0, vcBit.Length - 1);
            }
            vcBit = "vc://cosmos09/relevance";

            string remainingPath = cosmosLogPath.Substring(vcLength);
            if (!remainingPath.EndsWith("/"))
            {
                remainingPath += "/";
            }

            //remainingPath = String.Format("\"{0}LogDuringSetupEnvironment/DLIS/DLMSContainerOutputLog_.log\"", remainingPath);
            remainingPath = String.Format("\"{0}-1/query_output.log", remainingPath);
            File.WriteAllLines(localOutputPath, new string[] { remainingPath, vcBit });
        }

        /// <summary>
        /// Optionally records the module pass/fail state, and returns the return-code that this program should
        /// report when it exits, depending on how it is configured.
        /// </summary>
        /// <param name="returnCode">The "real" return code.</param>
        /// <param name="alwaysPass">If the module is configured to always pass.</param>
        /// <param name="passFailPath">Path to report pass/fail</param>
        /// <returns>The return code which this module should actually return.</returns>
        private static int RecordPassFail(int returnCode, bool alwaysPass, string passFailPath)
        {
            if (!string.IsNullOrEmpty(passFailPath))
            {
                File.WriteAllText(passFailPath, returnCode == 0 ? "true" : "false");
            }

            if (alwaysPass)
            {
                return 0;
            }
            else
            {
                return returnCode;
            }
        }


        /// <summary>
        /// Aether's default parameters are formatted a little differently than what the parser here expects. "param=value", instead of "--param value".  Parse out any of these that are wanted.
        /// </summary>
        /// <param name="cmdLineArgs"></param>
        /// <param name="aetherParameters"></param>
        /// <returns></returns>
        private static string[] ExtractAetherParameters(string[] cmdLineArgs, string[] aetherParameters)
        {
            string[] result = new string[aetherParameters.Length];
            for (int i = 0; i < aetherParameters.Length; i++)
            {
                result[i] = string.Empty;
                string key = aetherParameters[i].ToLower() + "=";
                string token = cmdLineArgs.Where(c => c.ToLower().StartsWith(key)).FirstOrDefault();
                if (token != null)
                {
                    result[i] = token.Substring(key.Length);
                }
            }

            return result;
        }

        /// <summary>
        /// For the model path and query path, Aether hands us pathes to files which contain these values, rather than the values themselves.
        /// If this string is a local path on the Aether worker disk, this function will return its contents.  Otherwise, it will return the original string back.
        /// </summary>
        /// <param name="inputString"></param>
        /// <returns></returns>
        private static string SubstituteLocalFileContentsIfNeeded(string inputString)
        {
            if (!File.Exists(inputString))
            {
                return inputString;
            }

            return File.ReadAllText(inputString).Trim();
        }
    }
}

