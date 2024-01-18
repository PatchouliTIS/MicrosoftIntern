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
    public class AggregateFinalResponsePlugin : IPlugin
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public PluginResult Execute(PluginServices pluginServices,
                                    PluginOutput<global::Platform.LegacyQueryResponseData> output,
                                    global::Platform.LegacyQueryResponseData baseData,
                                    global::Platform.LegacyQueryResponseData addData = null
                                    )
        {
            pluginServices.Logger.Info("ENTERING Aggregate");
            var outputData = pluginServices.CreateInstance<global::Platform.LegacyQueryResponseData>();

            outputData.LegacyAqr = baseData.LegacyAqr.LightClone();
            
            //foreach ( var item in baseData.LegacyAqr.ListAnswers.Elements) 
            //{
            //    outputData.LegacyAqr.ListAnswers.Elements.Add( item );
            //}
            if(addData == null)
            {
                pluginServices.Logger.Info("CreateBondAQR is Empty");
            } else
            {
                pluginServices.Logger.Info("Receiving CreateBondAQR output");
                if(addData.LegacyAqr == null)
                {
                    pluginServices.Logger.Info("addData.LegacyAqr is nll");
                    
                }else
                {
                    if (addData.LegacyAqr.ListAnswers == null)
                    {
                        pluginServices.Logger.Info("addData.LegacyAqr.ListAnswers is nll");
                    }else
                    {
                        if(addData.LegacyAqr.ListAnswers.Elements == null)
                        {
                            pluginServices.Logger.Info("addData.LegacyAqr.ListAnswers.Element is nll");
                        } else
                        {
                            if(addData.LegacyAqr.ListAnswers.Elements.First() == null)
                            {
                                pluginServices.Logger.Info("addData.LegacyAqr.ListAnswers.Element.First() is nll");
                            }else
                            {
                                foreach (var item in addData.LegacyAqr.ListAnswers.Elements)
                                {
                                    pluginServices.Logger.Info("ADDING outputdata");
                                    outputData.LegacyAqr.ListAnswers.Elements.Add(item);
                                }
                            }
                        }
                    }
                }
            }

            output.Data = outputData;


            pluginServices.Logger.Info(outputData.LegacyAqr.ListAnswers.Elements.ToString());

            return PluginResult.Succeeded;
        }
    }
}
