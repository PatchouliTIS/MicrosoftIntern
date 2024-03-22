using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.PluginFramework;

namespace CrossLangCache.Plugins
{
    public class QIConditionPlugin : IConditionPlugin
    {
        public const string SERVICE_NAME = "crosslangsearch";
        public const string SCENARIO_NAME = "firstpage";


        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public PluginConditionResult Execute(PluginServices pluginServices,
                                             global::QueryIndex.CanonicalQueriesOutputV3 preWebV3PrecisionOutput)
        {

            if(preWebV3PrecisionOutput == null)
            {
                pluginServices.Logger.Info("QI Output is NULL!");
            } else
            {
                if (preWebV3PrecisionOutput.RankedCanonicalQueries == null)
                {
                    pluginServices.Logger.Info("Canonical Query is Null");
                }
            }
            foreach (QueryIndex.CanonicalQueryCandidateV3 canonicalQuery in preWebV3PrecisionOutput.RankedCanonicalQueries)
            {
                string query = canonicalQuery.CanonicalQuery;      // expanded query QI output
                foreach (QueryIndex.AnswerContextV3 answer in canonicalQuery.AnswerContextsV3)
                {//each QueryIndex.AnswerContextV3 is an unit for the specific answer.
                    double QQSimilarityScore = answer.QQSimilarityScore;             // QQ SimilarityScore     
                    double annScore = answer.ANNSearchScore;             // ANN similarity score 
                    if (answer.Service == SERVICE_NAME.ToLower() && answer.Scenario == SCENARIO_NAME.ToLower())
                    {
                        if(QQSimilarityScore >= 0.9 && annScore >= 0.8)
                        {
                            pluginServices.Logger.Info("QueryIndex Triggered --> CrossLangSearch");
                            return new PluginConditionResult(true, true);
                        } else
                        {
                            pluginServices.Logger.Info("Failed QueryIndex Trigger --> QQSimilarityScore:{0}  annScore:{1}", QQSimilarityScore, annScore);
                            return new PluginConditionResult(false, false);
                        }
                    }

                }
            }
            pluginServices.Logger.Info("No CrossLangSearch Service Detected in QueryIndex Output");
            return new PluginConditionResult(false, false);
        }

    }
}
