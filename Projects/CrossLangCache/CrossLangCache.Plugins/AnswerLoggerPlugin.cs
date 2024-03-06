using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.PluginFramework;

namespace CrossLangCache.Plugins
{
    public class AnswerLoggerPlugin : IPlugin
    {


        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public PluginResult Execute(PluginServices pluginServices,
                                    PluginOutput<global::Platform.LegacyQueryResponseData> output,
                                    global::Platform.Query inputQuery = null,
                                    global::Platform.LegacyQueryResponseData input=null)
        {
            pluginServices.Logger.Info("ENTERING Aggregate");


            output.Data = pluginServices.CreateInstance<global::Platform.LegacyQueryResponseData>();
            output.Data = input;
            output.Data.LegacyAqr = input.LegacyAqr.LightClone();

            pluginServices.Logger.Info("Is Query Null:{0}", inputQuery == null);
            if(inputQuery != null)
            {
                pluginServices.Logger.Info("Query:{0}", inputQuery.RawQuery);
            }

            pluginServices.Logger.Info("Is LegacyQueryResponseData Null:{0}", input == null);

            var addData = output.Data;

            if (addData == null)
            {
                pluginServices.Logger.Info("CrossLangCache Answer -> LegacyQueryResponseData is Empty");
            }
            else
            {
                pluginServices.Logger.Info("Retrieving CrossLangCache Answer LegacyQueryResponseData");
                if (addData.LegacyAqr == null)
                {
                    pluginServices.Logger.Info("Answer.LegacyAqr is nll");

                }
                else
                {
                    if (addData.LegacyAqr.ListAnswers == null)
                    {
                        pluginServices.Logger.Info("Answer.LegacyAqr.ListAnswers is nll");
                    }
                    else
                    {
                        if (addData.LegacyAqr.ListAnswers.Elements == null)
                        {
                            pluginServices.Logger.Info("Answer.LegacyAqr.ListAnswers.Element is nll");
                        }
                        else
                        {
                            if (addData.LegacyAqr.ListAnswers.Elements.First() == null)
                            {
                                pluginServices.Logger.Info("Answer.LegacyAqr.ListAnswers.Element.First() is nll");
                            }
                            else
                            {
                                foreach (var item in addData.LegacyAqr.ListAnswers.Elements)
                                {
                                    pluginServices.Logger.Info("CURRENT DATA:\nLinkSourceUrl:{0}\nElementText:{1}\nLinkClickUrl:{2}\n", item.ElTitle.LinkSourceUrl, item.ElTitle.ElementText, item.ElTitle.LinkClickUrl);
                                }
                            }
                        }
                    }
                }
            }

            return PluginResult.Succeeded;
        }
    }
}
