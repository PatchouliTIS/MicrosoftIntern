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

namespace CrossLangCache.Plugins
{
    [MajorVersionForLipstick(1)]
    public class CacheAnswersRankingPlugin : IPlugin
    {
        private const string Service = "crosslangsearch";
        private const string Scenario = "firstpage";
        private const string Feed = "middle";
        private const string Feature = "crosslangcache";


        public PluginResult Execute(PluginServices pluginServices,
                                    [ConfigFile("CacheAnswersRankingConfig.ini")]
                                    Platform.BoolData CrosslangCachePlace,
                                    Platform.LegacyQueryResponseData WAinput,
                                    PluginOutput<AnswersRanker.RankedContent_1> MOP3CrossLangCacheKnobRankedContent,
                                    PluginOutput<Platform.LegacyQueryResponseData> MOP3CrossLangCacheReponseData)
        {
            pluginServices.Logger.Info("CrossLangCache AnswersRankingPlugin execute.");
            var finalLog = "CrossLangCache AnswersRankingPlugin complete.";

            bool placeCrosslangCache = CrosslangCachePlace != null && CrosslangCachePlace.Value;

            if (!placeCrosslangCache)
            {
                pluginServices.Logger.Info("Disabled in config file. No crosslangcache knob to be placed at MOP-3.");
                pluginServices.Logger.Info(finalLog);
                return new PluginResult(true);
            }



            pluginServices.Logger.Info("CrossLangCache to be placed at MOP-3. crosslangcache knob ranked content and ADO to be created.");

            var rankedContent = CreateCrossLangCacheRankedContentAtMOP3(pluginServices);
            MOP3CrossLangCacheKnobRankedContent.Data = rankedContent;
            pluginServices.Logger.Info("CrossLangCache knob ranked content created with Service: {0}, Scenario: {1}, AnswerId: {2}.", rankedContent.ServiceName, rankedContent.AnswerScenario, rankedContent.AnswerId);

            // use LegacyQueryResponseData directly
            var response = CreateCrossLangCacheReponseAtMOP3(pluginServices, WAinput, rankedContent.AnswerId);
            MOP3CrossLangCacheReponseData.Data = response;
            pluginServices.Logger.Info("CrossLangCache Response Data created");



            pluginServices.Logger.Info(finalLog);
            return new PluginResult(true);
        }



        [SuppressMessage(category: "XapBuildCodeAnalysis.CodeRestrictions", checkId: "XN123:Users of protected framework methods must apply for an exception.", Target = "AnswersRanker.RankedContent_1 CrossLangCache.Plugins.CacheAnswersRankingPlugin::CreateCrossLangCacheRankedContentAtMOP3(Xap.PluginFramework.PluginServices)", Justification = "{07b60061-d893-4d5c-bf29-fcf0b8abbd64} In order to use protected framework methods you must first receive an exception from the XAP team.")]
        public static AnswersRanker.RankedContent_1 CreateCrossLangCacheRankedContentAtMOP3(PluginServices pluginServices)
        {
            pluginServices.Logger.Verbose("CreateCrossLangCacheRankedContentAtMOP3 begin.");



            var lsrc = pluginServices.LegacyShimRequestContext;



            var rankedContent = pluginServices.CreateInstance<AnswersRanker.RankedContent_1>();
            rankedContent.ServiceName = Service;
            rankedContent.AnswerScenario = Scenario;
            rankedContent.AnswerFeed = Scenario;
            rankedContent.Confidence = 1;
            rankedContent.VirtualServiceName = Service;
            rankedContent.AnswerId = lsrc == null ? 0 : (uint)lsrc.GetNextLegacyAdoContextId();
            rankedContent.AnswerDesiredPosition = 7;
            rankedContent.AnswerOwnerAliases = "zhaotaipan";
            rankedContent.AnswerPossibleNamedPositions = new List<string> { "MOP-3" };
            rankedContent.ByPassDarwinRankers = true;
            rankedContent.IsMultiTurn = true;

            pluginServices.Logger.Verbose("Dummy RankedContent created with Service: {0}, Scenario: {1}, Feed: {2}, Confidence: {3}, VirtualService: {4}, AnswerId: {5}, AnswerDesiredPosition: {6}, AnswerPossibleNamedPositions: {7}, ByPassDarwinRankers: {8}, IsMultiTurn: {9}.",
                                                                            rankedContent.ServiceName, rankedContent.AnswerScenario, rankedContent.AnswerFeed, rankedContent.Confidence, rankedContent.VirtualServiceName, rankedContent.AnswerId, rankedContent.AnswerDesiredPosition, string.Join(" ", rankedContent.AnswerPossibleNamedPositions), rankedContent.ByPassDarwinRankers, rankedContent.IsMultiTurn);

            pluginServices.Logger.Verbose("CreateCrossLangCacheRankedContentAtMOP3 complete.");
            return rankedContent;
        }

        private static global::Platform.LegacyQueryResponseData CreateCrossLangCacheReponseAtMOP3(PluginServices pluginServices, Platform.LegacyQueryResponseData WAinput, uint answerid)
        {
            pluginServices.Logger.Verbose("CreateCrossLangCacheReponseAtMOP3 begin.");

            if (WAinput == null)
            {
                pluginServices.Logger.Info("INPUT webResponseAqr IS NULL!!");
            }

            // 1. copy from webResponseAqr
            var outputResponse = pluginServices.CreateInstance<global::Platform.LegacyQueryResponseData>();
            outputResponse.LegacyAqr = WAinput.LegacyAqr.LightClone();

            // 2. change Scenario, Service Name etc. in AnswerData
            if (outputResponse == null)
            {
                pluginServices.Logger.Info("CreateBondAQR is Empty");
            }
            else
            {
                pluginServices.Logger.Info("Receiving WebAnswer.webResponseAqr");
                if (outputResponse.LegacyAqr == null)
                {
                    pluginServices.Logger.Info("outputResponse.LegacyAqr is nll");

                }
                else
                {
                    if (outputResponse.LegacyAqr.ListAnswers == null)
                    {
                        pluginServices.Logger.Info("outputResponse.LegacyAqr.ListAnswers is nll");
                    }
                    else
                    {
                        if (outputResponse.LegacyAqr.ListAnswers.Elements == null)
                        {
                            pluginServices.Logger.Info("outputResponse.LegacyAqr.ListAnswers.Element is nll");
                        }
                        else
                        {
                            if (outputResponse.LegacyAqr.ListAnswers.Elements.First() == null)
                            {
                                pluginServices.Logger.Info("outputResponse.LegacyAqr.ListAnswers.Element.First() is nll");
                            }
                            else
                            {
                                foreach (var item in outputResponse.LegacyAqr.ListAnswers.Elements)
                                {
                                    pluginServices.Logger.Info("Modify AnswersData");
                                    item.AnswerScenario = Scenario;
                                    item.AnswerFeed = Feed;
                                    item.ServiceName = Service;
                                    item.UxDisplayHint = "KifResponse";
                                    item.SimpleDisplayText = "CrossLangCache LegacyQueryResponseData";
                                    item.IdInContext = answerid;
                                }
                            }
                        }
                    }
                }
            }

            pluginServices.Logger.Verbose("CreateCrossLangCacheReponseAtMOP3 complete.");
            return outputResponse;
        }
    }
}
