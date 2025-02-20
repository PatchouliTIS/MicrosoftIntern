﻿
using Platform;

namespace Xap
{
    using System;
    using System.Collections.Generic;
    using System.Net.WebSockets;
    using System.Text.Json;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;
    using global::Xap.ComponentFramework;
    using global::Xap.PluginFramework;
    using System.Linq;
    using System.Collections.Concurrent;
    using System.Security;
    using System.Runtime.InteropServices;
    using System.Net;
    using System.Diagnostics.CodeAnalysis;




    #region PluginClass
    // 32, 643
    [MajorVersionForLipstick(2)]
    [EnableLegacyCache(true)]
    [Timeout(@"*", 120000)]
    public class QueryModifyPlugin : IAsyncPlugin
    {
        [SuppressMessage(category:"XapBuildCodeAnalysis.Performance", checkId: "XN104:Certain Namespaces are not allowed in Xap", Target="System.String Xap.QueryModifyPlugin::ConvertToUnsecureString(System.Security.SecureString)", Justification = "{6becc20d-1d3e-46f8-939b-b1e2b0686443} Attempted use of restricted Namespace")]
        [SuppressMessage(category:"XapBuildCodeAnalysis.Performance", checkId: "XN104:Certain Namespaces are not allowed in Xap", Target="System.String Xap.QueryModifyPlugin::ConvertToUnsecureString(System.Security.SecureString)", Justification = "{9774b2a2-23f3-4b9b-ac8f-52a16b479f7a} Attempted use of restricted Namespace")]
        [SuppressMessage(category:"XapBuildCodeAnalysis.Performance", checkId: "XN104:Certain Namespaces are not allowed in Xap", Target="System.String Xap.QueryModifyPlugin::ConvertToUnsecureString(System.Security.SecureString)", Justification = "{3a3f3a9c-0f05-408e-b66b-c22504d259b6} Attempted use of restricted Namespace")]
        public static string ConvertToUnsecureString(SecureString secureString)
        {
            IntPtr unmanagedString = IntPtr.Zero;
            try
            {
                unmanagedString = Marshal.SecureStringToGlobalAllocUnicode(secureString);  
                return Marshal.PtrToStringUni(unmanagedString);
            }
            finally
            {
                Marshal.ZeroFreeGlobalAllocUnicode(unmanagedString);
            }
            
        }

        #region ExecuteMethod
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public async Task<PluginResult> Execute(PluginServices pluginServices,
                                    global::Platform.Query query,
                                    global::Platform.Augmentations augmentations,
                                    PluginOutput<global::Platform.Query> modifiedQuery)
        {
            modifiedQuery.Data = pluginServices.CreateInstance<global::Platform.Query>(query);


            IDictionary<string, string> crossLangParams = null;
            string mkt = "default";
            bool isUserAugmentation = true;
            if (!(augmentations == null || augmentations.Augmentation == null || !augmentations.Augmentation.Any()))
            {
                augmentations.Augmentation.TryGetValue("CrossLangParams", out crossLangParams);
                if (crossLangParams != null)
                {
                    if (crossLangParams.TryGetValue("targetLanguage", out string targetLanguage) && crossLangParams.TryGetValue("targetRegion", out string targetRegion))
                    {
                        mkt = $"{targetLanguage}-{targetRegion}";
                        pluginServices.Logger.Info("user augmentation mkt:  " + mkt);
                    }
                    else
                    {
                        isUserAugmentation = false;
                    }
                }
                else
                {
                    isUserAugmentation = false;
                }
            }
            else
            {
                isUserAugmentation = false;
            }

            if (!isUserAugmentation && pluginServices.Variants.TryGetValue("MKT", out mkt))
            {
                pluginServices.Logger.Info("MKT:  " + mkt);
            }

            /** .Data 就是当前的 Platform.Query 其中包含了一系列的内容：
             *     string RawQuery
             *     string NormalizedQuery
             *     string WordBrokenQuery
             *     String2StringMapping WordBrokenToRawQueryMappin
             *     QueryLanguage QueryLanguage
             * **/

            // 0. check the language

            mkt = mkt.ToLower();

            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict["zh-cn"] = "Chinese";
            dict["en-us"] = "English";
            dict["ja-jp"] = "Japanese";
            dict["de-de"] = "German";
            dict["fr-fr"] = "French";
            dict["es-es"] = "Spanish";
            dict["default"] = "English";

            // 1. set the prompt
            string prompt = $"You are translator. You are also search engine expert. Output the translation of the query into {dict[mkt]}.\r\nQuery: ";
            prompt += query.RawQuery + "\r\nTranslation:";

            pluginServices.Logger.Info(prompt);

            // 2. decode the token from .ini(base64)
            SecureString codexToken;
            if (pluginServices.Secrets.TryGetSecret("wxtcstrainkv", "codex-playground-token", out codexToken))
            {
                string decodedCodexToken = ConvertToUnsecureString(codexToken);

                CodexGetter getter = new CodexGetter(prompt, 0.7, 64, 1.0, true, decodedCodexToken, pluginServices);
                Task getterTask =  getter.Connect();
                getterTask.Wait();

                pluginServices.Logger.Info(">>>WAITING FOR DUTY<<<");
                string messages = "";
                bool mark = false;

                while (!getter.Messages.IsEmpty)
                {
                    // pluginServices.Logger.Info("Receiving Data");
                    // messages = "";

                    if (getter.Messages.TryDequeue(out string message))
                    {
                        if (message == null)
                        {
                            pluginServices.Logger.Info("current message is empty");
                            continue;
                        }
                        else
                        {
                            pluginServices.Logger.Info("Processing Message Cut");
                            if (message.Contains("\n") || message.Contains("\r\n"))
                            {
                                int idx = message.IndexOf("\n");
                                if (idx == -1)
                                {
                                    idx = message.IndexOf("\r\n");
                                }

                                message = message.Substring(0, idx);
                                mark = true;
                            }

                            pluginServices.Logger.Info("current msg:" + message);
                            messages += message;



                            if(mark)
                            {
                                pluginServices.Logger.Info("WebSocket Done");
                                break;
                            }
                        }
                    }
                }


                if (!string.IsNullOrEmpty(messages))
                {
                    modifiedQuery.Data.RawQuery = messages;
                    modifiedQuery.Data.NormalizedQuery = messages;
                }

                pluginServices.Logger.Info(">>> Query Modified Done <<<");
                pluginServices.Logger.Info("Modified Query:" + modifiedQuery.Data.RawQuery);

            }
            else
            {
                pluginServices.Logger.Info(">>>KEYVAULT TOKEN NOT FOUND<<<");
            }
            return PluginResult.Succeeded;
        }
        #endregion ExecuteMethod



        #endregion PluginClass






        #region CodexGetter
        class CodexGetter
        {
            private ConcurrentQueue<string> messages = new ConcurrentQueue<string>();
            private string prompt;
            private double temperature;
            private int maxTokens;
            private double topP;
            private bool streaming;
            private string gptToken;
            private PluginServices pluginServices;


            public ConcurrentQueue<string> Messages { get { return messages; } }

            public CodexGetter(string prompt, double temperature = 0.7, int maxTokens = 256, double topP = 1, bool streaming = false, string gptToken = null, PluginServices pluginServices = null)
            {
                this.prompt = prompt;
                this.temperature = temperature;
                this.maxTokens = maxTokens;
                this.topP = topP;
                this.streaming = streaming;
                this.gptToken = gptToken;
                this.pluginServices = pluginServices;
            }

            public async Task ReceiveData(ClientWebSocket websocket)
            {
                var buffer = new ArraySegment<byte>(new byte[1024]);
                while (websocket.State == WebSocketState.Open)
                {
                    var result = await websocket.ReceiveAsync(buffer, CancellationToken.None);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        break;
                    }

                    var message = Encoding.UTF8.GetString(buffer.Array, buffer.Offset, result.Count);
                    var jsonDocument = JsonDocument.Parse(message);
                    var root = jsonDocument.RootElement;

                    if (root.TryGetProperty("type", out var typeProperty))
                    {
                        var type = typeProperty.GetString();
                        switch (type)
                        {
                            case "status":
                                if (root.TryGetProperty("details", out var detailsProperty) && detailsProperty.GetString() == "Done")
                                {
                                    return;
                                }
                                break;
                            case "sse":
                                var choicesArray = root.GetProperty("payload").GetProperty("choices").EnumerateArray();
                                if (choicesArray.Any())
                                {
                                    var text = choicesArray.First().GetProperty("text").GetString();
                                    pluginServices.Logger.Info("CodexGetter Text:" + text);

                                    if (!streaming)
                                    {
                                        messages.Clear(); // Clear the queue if not in streaming mode
                                    }
                                    messages.Enqueue(text);
                                }
                                break;
                        }
                    }
                }
            }

            public async Task SendData(ClientWebSocket websocket, CancellationToken cancellationToken)
            {
                // string decodeToken = Encoding.Default.GetString(Convert.FromBase64String(this.gptToken));
                var payload = new
                {
                    prompt = this.prompt,
                    temperature = this.temperature,
                    max_tokens = this.maxTokens,
                    top_p = this.topP
                };

                var message = new
                {
                    endpoint = "chatgpt",
                    token = (object)null,
                    auth = this.gptToken,
                    custom_headers = new Dictionary<string, object>(),
                    payload = payload
                };

                var jsonMessage = JsonSerializer.Serialize(message);
                var data = Encoding.Default.GetBytes(jsonMessage);

                // Send a message to the server
                await websocket.SendAsync(new ArraySegment<byte>(data), WebSocketMessageType.Text, true, cancellationToken);
            }

            public async Task Connect()
            {
                using (var websocket = new ClientWebSocket())
                {
                    var cancellationTokenSource = new CancellationTokenSource();
                    var uri = new Uri("wss://codexplayground.azurewebsites.net/papyrus/");

                    await websocket.ConnectAsync(uri, CancellationToken.None);
                    var receiveTask = ReceiveData(websocket);
                    await SendData(websocket, CancellationToken.None);
                    await receiveTask;
                }
            }

        }
        #endregion
    }
}



