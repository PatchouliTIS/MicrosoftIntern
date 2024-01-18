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


    [MajorVersionForLipstick(1)]
    [EnableLegacyCache(true)]
    [Timeout(@"*", 9000)]
    public class DisplayResultBondPlugin : IAsyncPlugin
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public Task<PluginResult> Execute(PluginServices pluginServices,
                                    global::Platform.Query query,
                                    PluginOutput<CrossLang.QueryDisplay> output,
                                    PluginOutput<global::Platform.StringData> serviceName,
                                    PluginOutput<global::Platform.StringData> scenarioName)
        {
            var queryBond = pluginServices.CreateInstance<CrossLang.QueryDisplay>();

            queryBond.TargetQuery = query.RawQuery;
            

            if(pluginServices.Variants.TryGetValue("MKT", out string mkt))
            {
                pluginServices.Logger.Info("Get MKT Succuss:" + mkt);
                queryBond.TargetMarket = mkt;
            } else
            {
                pluginServices.Logger.Info("Get MKT Failed: No MKT exists");
            }

            if (pluginServices.Variants.TryGetValue("LANGUAGE", out string lang))
            {
                pluginServices.Logger.Info("Get LANGUAGE Succuss:" + lang);
                queryBond.TargetLanguage = lang;
            }
            else
            {
                pluginServices.Logger.Info("Get LANGUAGE Failed: No LANGUAGE exists");
            }


            output.Data = queryBond;


            serviceName.Data = pluginServices.CreateInstance<global::Platform.StringData>();
            serviceName.Data.Value = @"BingCrossLang";

            scenarioName.Data = pluginServices.CreateInstance<global::Platform.StringData>();
            scenarioName.Data.Value = @"QueryResponse";




            return Task.FromResult(PluginResult.Succeeded);
        }
    }
}
