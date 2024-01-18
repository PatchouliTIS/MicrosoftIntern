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
                                    PluginOutput<global::Platform.Augmentations> modifiedAugmentations)
        {
            modifiedQuery.Data = pluginServices.CreateInstance<global::Platform.Query>(query);
            modifiedAugmentations.Data = pluginServices.CreateInstance<global::Platform.Augmentations>(augmentations);

            // modify the variants via Platform.Augmentations


            if (!pluginServices.Variants.TryGetValue("MKT", out string value))
            {
                // Console.WriteLine("No MKT Variant exist");
                pluginServices.Logger.Info("No MKT Variant exist");
               
            } else
            {
                // Console.WriteLine("MKT:  " + value);
                pluginServices.Logger.Info("MKT:  " + value);
            }

            if (!pluginServices.Variants.TryGetValue("TARGETMKT", out string tgt))
            {
                // Console.WriteLine("No MKT Variant exist");
                pluginServices.Logger.Info("No TARGETMKT Variant exist");

            }
            else
            {
                // Console.WriteLine("MKT:  " + tgt);
                pluginServices.Logger.Info("TARGETMKT:  " + tgt);
            }

            // Console.WriteLine(">>>> ENTERING AUGMENTATION <<<<");
            pluginServices.Logger.Info(">>>> ENTERING AUGMENTATION <<<<");
            IDictionary<string, string> section = null;
            if (!(augmentations == null || augmentations.Augmentation == null || !augmentations.Augmentation.Any()))
            {
                foreach(var item in augmentations.Augmentation)
                {
                    // Console.WriteLine($"{item}");
                    pluginServices.Logger.Info($"{item.Key}");
                    foreach(var item2 in item.Value)
                    {
                        pluginServices.Logger.Info($"{item2}");
                    }
                }

                augmentations.Augmentation.TryGetValue("TargetAug", out section);

            } else
            {
                // Console.WriteLine("Augmentations is EMPTY!!!");
                pluginServices.Logger.Info("Augmentations is EMPTY!!!");
            }

            if (section == null) 
            {
                pluginServices.Logger.Info("TargetAug is empty");
            } else
            {
                if (section.TryGetValue("mkt", out string mkt))
                {
                    pluginServices.Logger.Info("augmentation's mkt ---->" + mkt);
                }
                else
                {
                    pluginServices.Logger.Info("augmentation's mkt is empty");
                }
                if (section.TryGetValue("tmkt", out string targetmkt))
                {
                    pluginServices.Logger.Info("augmentation's tmkt ---->" + targetmkt);
                }
                else
                {
                    pluginServices.Logger.Info("augmentation's tmkt is empty");
                }
            }
            

            return Task.FromResult(PluginResult.Succeeded);
        }
    }
}
