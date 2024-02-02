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
    using Xap.WorkflowFramework;

    [MajorVersionForLipstick(1)]
    [EnableLegacyCache(true)]
    [Timeout(@"*", 9000)]
    public class VariantSetPlugin : IAsyncPlugin
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public Task<PluginResult> Execute(PluginServices pluginServices,
                                    global::Platform.Query query,
                                    global::Platform.Augmentations augmentations,
                                    PluginOutput<global::Platform.Query> modifiedQuery,
                                    PluginOutput<global::Platform.Augmentations> modifiedAugmentations,
                                    PluginOutput<global::Platform.LegacyPluginAugmentsAndVariantsData> modifiedUserAugmentations)
        {
            modifiedQuery.Data = pluginServices.CreateInstance<global::Platform.Query>(query);
            modifiedAugmentations.Data = pluginServices.CreateInstance<global::Platform.Augmentations>(augmentations);

            // modify the variants via Platform.Augmentation

            if (!pluginServices.Variants.TryGetValue("MKT", out string value))
            {
                pluginServices.Logger.Info("No MKT Variant exist");
               
            } else
            {
                pluginServices.Logger.Info("MKT:  " + value);
            }

            if (!pluginServices.Variants.TryGetValue("TARGETMKT", out string tgt))
            {
                pluginServices.Logger.Info("No TARGETMKT Variant exist");
            }
            else
            {
                pluginServices.Logger.Info("TARGETMKT:  " + tgt);
            }

            pluginServices.Logger.Info(">>>> ENTERING AUGMENTATION <<<<");
            IDictionary<string, string> crossLangParams = null;
            if (!(augmentations == null || augmentations.Augmentation == null || !augmentations.Augmentation.Any()))
            {
                augmentations.Augmentation.TryGetValue("CrossLangParams", out crossLangParams);

            } else
            {
                pluginServices.Logger.Info("Augmentations is EMPTY!!!");
            }

            var overrideAugementations = pluginServices.CreateInstance<global::Platform.LegacyPluginAugmentsAndVariantsData>();
            if (crossLangParams == null) 
            {
                pluginServices.Logger.Info("CrossLangParams is empty");   
                overrideAugementations.Augmentation = @"[AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]";
                modifiedUserAugmentations.Data = overrideAugementations;
            } else
            {
                if (crossLangParams.TryGetValue("mkt", out string mkt))
                {
                    pluginServices.Logger.Info("augmentation's mkt ---->" + mkt);
                }
                else
                {
                    pluginServices.Logger.Info("augmentation's mkt is empty");
                }
                if (crossLangParams.TryGetValue("targetLanguage", out string targetLanguage))
                {
                    pluginServices.Logger.Info("augmentation's targetLanguage ---->" + targetLanguage);
                }
                else
                {
                    pluginServices.Logger.Info("augmentation's targetLanguage is empty");
                    overrideAugementations.Augmentation = @"[AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]";
                    modifiedUserAugmentations.Data = overrideAugementations;
                }
                if (crossLangParams.TryGetValue("targetRegion", out string targetRegion))
                {
                    pluginServices.Logger.Info("augmentation's targetRegion ---->" + targetRegion);
                    // for zh language, it will need to map to zh-hans
                    string customAugmentations = "";
                    if (targetLanguage == "zh")
                    {
                        customAugmentations = $"[AdService EnableORV=\"1\"][BAH Workflow=\"Workflow_APlusNext\"][UFLanguage UnderstandLangs=\"zh-hans, en\"]";
                    }
                    else
                    {
                        customAugmentations = $"[AdService EnableORV=\"1\"][BAH Workflow=\"Workflow_APlusNext\"][UFLanguage UnderstandLangs=\"{targetLanguage}, en\"]";
                    } 
                    // the mkt, truemkt and region will decide the web result
                    string customVariants = String.Format("mkt:{0}-{1}&truemkt:{0}-{1}&region:{1}", targetLanguage, targetRegion);
                    pluginServices.Logger.Info("override augmentation's variants ---->" + customVariants);
                    overrideAugementations.Augmentation = customAugmentations;
                    overrideAugementations.VariantConstraint = customVariants;
                    modifiedUserAugmentations.Data = overrideAugementations;
                }
                else
                {
                    pluginServices.Logger.Info("augmentation's targetRegion is empty");
                    overrideAugementations.Augmentation = @"[AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]";
                    modifiedUserAugmentations.Data = overrideAugementations;
                }
            }
            

            return Task.FromResult(PluginResult.Succeeded);
        }
    }
}
