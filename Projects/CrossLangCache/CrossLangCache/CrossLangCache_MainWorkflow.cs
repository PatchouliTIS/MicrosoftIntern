using AutosAnswer.Plugins;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.ComponentFramework;
using Xap.WorkflowFramework;

namespace Xap.Shared.CrossLang
{
    public class CrossLangCache_MainWorkflow : global::Xap.WorkflowFramework.Workflow
    {


        #region InputFields
        protected Task<global::Platform.Query> Query;
        protected Task<global::Platform.UserIdentification> UserIdentification;
        protected Task<global::Platform.InstrumentationData> InstrumentationData;
        protected Task<global::Platform.Augmentations> Augmentations;
        protected Task<global::Local.LocationContext_1> LocationContext;
        protected Task<global::Cortana.CU.RequestData> cuRequestData;
        protected Task<global::QueryIndex.CanonicalQueriesOutputV3> preWebV3PrecisionOutput;
        #endregion


        #region OutputFields
        protected Task<global::Platform.LegacyQueryResponseData> outputData;
        protected Task<AnswersRanker.RankedContent_1> outputRanker;
        #endregion



        public void ExecuteForInheritance()
        {
            this.Execute(
                outputData: out this.outputData,
                outputRanker: out this.outputRanker,
                Query: this.Query,
                UserIdentification: this.UserIdentification,
                InstrumentationData: this.InstrumentationData,
                Augmentations: this.Augmentations,
                LocationContext: this.LocationContext,
                cuRequestData: this.cuRequestData,
                preWebV3PrecisionOutput: this.preWebV3PrecisionOutput);
        }

        public void Execute(
            out Task<global::Platform.LegacyQueryResponseData> outputData,
            out Task<AnswersRanker.RankedContent_1> outputRanker,
            Task<global::Platform.Query> Query = null,
            Task<global::Platform.UserIdentification> UserIdentification = null,
            Task<global::Platform.InstrumentationData> InstrumentationData = null,
            Task<global::Platform.Augmentations> Augmentations = null,
            Task<global::Local.LocationContext_1> LocationContext = null,
            Task<global::Cortana.CU.RequestData> cuRequestData = null,
            Task<global::QueryIndex.CanonicalQueriesOutputV3> preWebV3PrecisionOutput = null)
        {
            this.Query = Query;
            this.UserIdentification = UserIdentification;
            this.InstrumentationData = InstrumentationData;
            this.Augmentations = Augmentations;
            this.LocationContext = LocationContext;
            this.cuRequestData = cuRequestData;
            this.preWebV3PrecisionOutput = preWebV3PrecisionOutput;

            this.BindQIConditionPlugin();
            AsyncIf(this.QIConditionPlugin.ExecuteForInheritance()).Then(() =>
            {

                this.BindCrossLangAugmentPlugin();
                this.CrossLangAugmentPlugin.ExecuteForInheritance();

                this.BindCacheAnswersRankingPlugin();
                this.CacheAnswersRankingPlugin.ExecuteForInheritance();

                this.BindCacheQueryPlugin();
                this.CacheQueryPlugin.ExecuteForInheritance();

                this.BindCacheResultPlugin();
                this.CacheResultPlugin.ExecuteForInheritance();

                this.BindAnswersRanking();
                this.AnswersRanking.ExecuteForInheritance();

                this.outputData = this.AnswersRanking.outputResponseData;
                this.outputRanker = this.CacheAnswersRankingPlugin.rankedContent;

            });


            outputData = this.outputData;
            outputRanker = this.outputRanker;

        }




        #region CrossLangAugmentationsPlugin
        [Timeout("*", 1000)]
        public CrossLangCache.Plugins.ICrossLangAugmentPlugin_ForInheritance CrossLangAugmentPlugin { get; } = WorkflowServices.CreateInstance<CrossLangCache.Plugins.ICrossLangAugmentPlugin_ForInheritance>();
        protected virtual void BindCrossLangAugmentPlugin()
        {
            this.CrossLangAugmentPlugin.Inputs.query = this.Query;
            this.CrossLangAugmentPlugin.Inputs.augmentations = this.Augmentations;
        }
        #endregion


        #region CacheQueryPlugin
        [Timeout("*", 1000)]
        public virtual CrossLangCache.Plugins.ICacheQueryPlugin_ForInheritance CacheQueryPlugin { get; } = WorkflowServices.CreateInstance<CrossLangCache.Plugins.ICacheQueryPlugin_ForInheritance>();
        protected virtual void BindCacheQueryPlugin()
        {
            this.CacheQueryPlugin.Inputs.preWebV3PrecisionOutput = this.preWebV3PrecisionOutput;
            this.CacheQueryPlugin.Inputs.targetMKT = this.CrossLangAugmentPlugin.targetMKT;
        }
        #endregion CacheQueryPlugin

        #region CacheAnswersRankingPlugin
        public virtual CrossLangCache.Plugins.ICacheAnswersRankingPlugin_ForInheritance CacheAnswersRankingPlugin { get; } = WorkflowServices.CreateInstance<CrossLangCache.Plugins.ICacheAnswersRankingPlugin_ForInheritance>();
        protected void BindCacheAnswersRankingPlugin()
        {
        }
        #endregion


        #region CacheResultPlugin
        public virtual CrossLangCache.Plugins.ICacheResultPlugin_ForInheritance CacheResultPlugin { get; } = WorkflowServices.CreateInstance<CrossLangCache.Plugins.ICacheResultPlugin_ForInheritance>();

        protected void BindCacheResultPlugin()
        {
            this.CacheResultPlugin.Inputs.Url = this.CacheQueryPlugin.Url;
            this.CacheResultPlugin.Inputs.Title = this.CacheQueryPlugin.Title;
            this.CacheResultPlugin.Inputs.Snippet = this.CacheQueryPlugin.Snippet;
        }
        #endregion


        #region CrossLangCache_AnswersRanking
        public virtual ICrossLangCache_AnswersRanking_ForInheritance AnswersRanking { get; } = WorkflowServices.CreateInstance<ICrossLangCache_AnswersRanking_ForInheritance>();

        protected void BindAnswersRanking()
        {
            this.AnswersRanking.Inputs.inputADO = this.CacheResultPlugin.MOP3CrossLangCacheADO;
            this.AnswersRanking.Inputs.answerId = this.CacheAnswersRankingPlugin.answerId;
        }
        #endregion




        #region QIConditionPlugin
        public virtual CrossLangCache.Plugins.IQIConditionPlugin_ForInheritance QIConditionPlugin { get; } = WorkflowServices.CreateInstance<CrossLangCache.Plugins.IQIConditionPlugin_ForInheritance>();
        protected void BindQIConditionPlugin()
        {
            this.QIConditionPlugin.Inputs.preWebV3PrecisionOutput = this.preWebV3PrecisionOutput;
        }
        #endregion




    }
}