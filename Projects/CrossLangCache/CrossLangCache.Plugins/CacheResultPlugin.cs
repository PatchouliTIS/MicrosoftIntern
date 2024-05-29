using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.ComponentFramework;
using Xap.PluginFramework;
using Platform;
using Entities.Containment;
using Kif2Bond.WholePageCache;
using Xap.AnswersWireFormat;
using System.Diagnostics.CodeAnalysis;
using LanguageGeneration;
using Kif;
using Kif2Bond.Kif;
using PD.Travel.TrainBook;

namespace CrossLangCache.Plugins
{
    public class CacheResultPlugin : IPlugin
    {

        public const string SERVICE = "crosslangsearch";
        public const string SCENARIO = "firstpage";
        public const string FEED = "middle";


        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public PluginResult Execute(PluginServices pluginServices,
                                    CollectionPluginOutput<IEnumerable<Platform.ADOContainer_1>> MOP3CrossLangCacheADO,
                                    IEnumerable<global::Platform.StringData> Url = null,
                                    IEnumerable<global::Platform.StringData> Title = null,
                                    IEnumerable<global::Platform.StringData> Snippet = null)
        {
            var outputApr = pluginServices.CreateInstance<Kif2Bond.Kif.AnswerProviderResponse_1>();
            outputApr.Results = pluginServices.CreateInstance<ICollection<Xap.Identity>>();
            outputApr.ClickThroughUrl = "Empty";
            outputApr.SubVertical = "Cross Language Search";

            var outputADO = pluginServices.CreateInstance<ADOContainer_1>();

            int len = Url.Count();

            if(len != Title.Count() || len != Snippet.Count())
            {
                return PluginResult.Failed("TOP-N: N is unmatched!");
            }

            for(int i = 0; i < len; ++i)
            {
                var outputWr = pluginServices.CreateInstance<Kif2Bond.WebAnswer.WebResult_1>();
                outputWr.Url = Url.ElementAt(i).Value;
                outputWr.Snippet = Snippet.ElementAt(i).Value;
                outputWr.Title = Title.ElementAt(i).Value;
                outputApr.Results.Add(outputWr);
                pluginServices.Logger.Info("URL:{0} Title:{1} Snippet:{2}\n", outputWr.Url, outputWr.Title, outputWr.Snippet);
            }


            outputADO.Result = outputApr;
            outputADO.ScenarioName = SCENARIO;
            outputADO.ServiceName = SERVICE;
            outputADO.AnswerFeed = FEED;
            outputADO.UxDisplayHint = "KifResponse";


            MOP3CrossLangCacheADO.Data = new List<global::Platform.ADOContainer_1> { outputADO };

            pluginServices.Logger.Verbose("CreateCrossLangCacheADO At MOP3 complete.");


            return PluginResult.Succeeded;
        }


    }



}
