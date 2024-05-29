using Ads.PaidSearchForNative;
using CrossLangCache.Plugins;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.WorkflowFramework;

namespace Xap.Shared.CrossLang
{
    public class CrossLangCache_AnswersRanking : global::Xap.WorkflowFramework.Workflow
    {
        #region InputFields
        protected Task<IEnumerable<Platform.ADOContainer_1>> inputADO;
        protected Task<IEnumerable<Platform.Int32Data>> answerId;
        #endregion


        #region OutputFields
        protected Task<global::Platform.LegacyQueryResponseData> outputResponseData;
        #endregion



        public void ExecuteForInheritance()
        {
            this.Execute(
                outputResponseData: out this.outputResponseData,
                inputADO: this.inputADO,
                answerId: this.answerId);
        }

        public void Execute(
            out Task<global::Platform.LegacyQueryResponseData> outputResponseData,
            Task<IEnumerable<Platform.ADOContainer_1>> inputADO = null,
            Task<IEnumerable<Platform.Int32Data>> answerId = null)
        {
            this.inputADO = inputADO ?? throw new ArgumentNullException("inputADO is NULL");
            this.answerId = answerId ?? throw new ArgumentNullException("answerId is NULL");

            this.ExecCreateAQR();
            outputResponseData = this.AQR.legacyQueryResponseData;

        }


        #region CreateAQR
        public virtual global::AHAPlusCommon.ICreateAQRFromList_3 AQR { get; } = WorkflowServices.CreateInstance<global::AHAPlusCommon.ICreateAQRFromList_3>();

        protected void ExecCreateAQR()
        {
            this.AQR.Execute(
                adoContainerList: this.inputADO,
                answerIdList: this.answerId);
        }
        #endregion



    }
}
