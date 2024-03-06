using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.PluginFramework;

namespace CrossLangCache.Plugins
{
    public class CrossLangAugmentPlugin : IPlugin
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public PluginResult Execute(PluginServices pluginServices,
                                    PluginOutput<global::Platform.Query> modifiedQuery,
                                    PluginOutput<global::Platform.Augmentations> modifiedAugmentations,
                                    PluginOutput<global::Platform.LegacyPluginAugmentsAndVariantsData> modifiedUserAugmentations,
                                    PluginOutput<global::Platform.StringData> targetMKT,
                                    global::Platform.Query query = null,
                                    global::Platform.Augmentations augmentations = null)
        {
            pluginServices.Logger.Info("Is BaseQuery Null:{0}", query == null);
            pluginServices.Logger.Info("Is BaseQuery String Null:{0}", String.IsNullOrEmpty(query.RawQuery));
            modifiedQuery.Data = query;
            pluginServices.Logger.Info("Is modifiedQuery Null:{0}", modifiedQuery.Data == null);
            pluginServices.Logger.Info("Is modifiedQuery String Null:{0}", String.IsNullOrEmpty(modifiedQuery.Data.RawQuery));
            modifiedAugmentations.Data = augmentations;
            targetMKT.Data = pluginServices.CreateInstance<global::Platform.StringData>();
            targetMKT.Data.Value = "zh-cn";


            // modify the variants via Platform.Augmentation
            if (!pluginServices.Variants.TryGetValue("MKT", out string value))
            {
                pluginServices.Logger.Info("No MKT Variant exist");

            }
            else
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

            }
            else
            {
                pluginServices.Logger.Info("Augmentations is EMPTY!!!");
            }

            var overrideAugementations = pluginServices.CreateInstance<global::Platform.LegacyPluginAugmentsAndVariantsData>();
            if (crossLangParams == null)
            {
                pluginServices.Logger.Info("CrossLangParams is empty");
                overrideAugementations.Augmentation = @"[AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]";
                modifiedUserAugmentations.Data = overrideAugementations;
            }
            else
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
                    targetMKT.Data.Value = String.Format("{0}-{1}", targetLanguage, targetRegion);
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




            return PluginResult.Succeeded;
        }


    }
}
