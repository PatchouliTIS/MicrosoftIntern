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
                                    CollectionPluginOutput<IEnumerable<Platform.Int32Data>> answerId,
                                    PluginOutput<AnswersRanker.RankedContent_1> rankedContent)
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



            // get answerId & RankedContent
            rankedContent.Data = CreateCrossLangCacheRankedContentAtMOP3(pluginServices);
            var id = pluginServices.CreateInstance<Platform.Int32Data>();
            id.Value = (int)rankedContent.Data.AnswerId;
            answerId.Data = new List<Platform.Int32Data> { id };
            pluginServices.Logger.Info("CrossLangCache Answer ID Generated");


            pluginServices.Logger.Info(finalLog);
            return new PluginResult(true);
        }

        [SuppressMessage("XapBuildCodeAnalysis.CodeRestrictions", "XN123:Users of protected framework methods must apply for an exception.", Target = "AnswersRanker.RankedContent_1 CrossLangCache.Plugins.CacheAnswersRankingPlugin::CreateCrossLangCacheRankedContentAtMOP3(Xap.PluginFramework.PluginServices)", Justification = "{07b60061-d893-4d5c-bf29-fcf0b8abbd64} In order to use protected framework methods you must first receive an exception from the XAP team.")]
        public static AnswersRanker.RankedContent_1 CreateCrossLangCacheRankedContentAtMOP3(PluginServices pluginServices)
        {
            pluginServices.Logger.Info("CreateCrossLangCacheRankedContentAtMOP3 begin.");
            ILegacyShimRequestContext legacyShimRequestContext = pluginServices.LegacyShimRequestContext;
            uint var = (legacyShimRequestContext != null) ? ((uint)legacyShimRequestContext.GetNextLegacyAdoContextId()) : 0u;

            var rankedContent = pluginServices.CreateInstance<AnswersRanker.RankedContent_1>();
            rankedContent.ServiceName = Service;
            rankedContent.AnswerScenario = Scenario;
            rankedContent.AnswerFeed = Feed;
            rankedContent.Confidence = 1;
            rankedContent.VirtualServiceName = Service;
            rankedContent.AnswerId = var;
            rankedContent.AnswerDesiredPosition = 7;
            rankedContent.AnswerOwnerAliases = "zhaotaipan";
            rankedContent.AnswerPossibleNamedPositions = new List<string> { "MOP-3" };
            rankedContent.ByPassDarwinRankers = true;
            rankedContent.IsMultiTurn = true;
            rankedContent.SuppressMe = false;

            pluginServices.Logger.Info("Dummy RankedContent created with Service: {0}, Scenario: {1}, Feed: {2}, Confidence: {3}, VirtualService: {4}, AnswerId: {5}, AnswerDesiredPosition: {6}, AnswerPossibleNamedPositions: {7}, ByPassDarwinRankers: {8}, IsMultiTurn: {9}.",
                                                                            rankedContent.ServiceName, rankedContent.AnswerScenario, rankedContent.AnswerFeed, rankedContent.Confidence, rankedContent.VirtualServiceName, rankedContent.AnswerId, rankedContent.AnswerDesiredPosition, string.Join(" ", rankedContent.AnswerPossibleNamedPositions), rankedContent.ByPassDarwinRankers, rankedContent.IsMultiTurn);

            pluginServices.Logger.Info("CreateCrossLangCacheRankedContentAtMOP3 complete.");
            return rankedContent;
        }

    }
}
