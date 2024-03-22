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
                                    global::Platform.LegacyQueryResponseData input = null,
                                    global::Platform.LegacyQueryResponseData WAoutput = null)
        {
            pluginServices.Logger.Info("ENTERING Aggregate");

            if (input == null)
            {
                pluginServices.Logger.Info("INPUT DATA IS NULL!!");
            }

            output.Data = pluginServices.CreateInstance<global::Platform.LegacyQueryResponseData>();
            output.Data = input;
            output.Data.LegacyAqr = input.LegacyAqr.LightClone();

            pluginServices.Logger.Info("Is Query Null:{0}", inputQuery == null);
            if (inputQuery != null)
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
                                    pluginServices.Logger.Info("CURRENT AnswerData:    Uri:{0}      ServiceName:{1}      KifResponse:{2}      KifResponseLength:{3}", item.Uri, item.ServiceName, item.KifResponseSegment.Array, item.KifResponseSegment.Count);
                                }
                            }
                        }
                    }
                }
            }



            if (WAoutput == null)
            {
                pluginServices.Logger.Info("WAoutput DATA IS NULL!!");
            }


            pluginServices.Logger.Info("Is LegacyQueryResponseData Null:{0}", WAoutput == null);



            if (WAoutput == null)
            {
                pluginServices.Logger.Info("WAoutput Answer -> LegacyQueryResponseData is Empty");
            }
            else
            {
                pluginServices.Logger.Info("Retrieving WAoutput Answer LegacyQueryResponseData");
                if (WAoutput.LegacyAqr == null)
                {
                    pluginServices.Logger.Info("WAoutput Answer.LegacyAqr is nll");

                }
                else
                {
                    if (WAoutput.LegacyAqr.ListAnswers == null)
                    {
                        pluginServices.Logger.Info("WAoutput Answer.LegacyAqr.ListAnswers is nll");
                    }
                    else
                    {
                        if (WAoutput.LegacyAqr.ListAnswers.Elements == null)
                        {
                            pluginServices.Logger.Info("WAoutput Answer.LegacyAqr.ListAnswers.Element is nll");
                        }
                        else
                        {
                            if (WAoutput.LegacyAqr.ListAnswers.Elements.First() == null)
                            {
                                pluginServices.Logger.Info("WAoutput Answer.LegacyAqr.ListAnswers.Element.First() is nll");
                            }
                            else
                            {
                                foreach (var item in WAoutput.LegacyAqr.ListAnswers.Elements)
                                {
                                    pluginServices.Logger.Info("WAoutput CURRENT AnswerData:    Uri:{0}      ServiceName:{1}      KifResponse:{2}      KifResponseLength:{3}", item.Uri, item.ServiceName, item.KifResponseSegment.Array, item.KifResponseSegment.Count);
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
