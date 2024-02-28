using AutosAnswer.Plugins;
using CrossLangCache.Plugins;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.ComponentFramework;
using Xap.WorkflowFramework;

namespace Xap.Shared.CrossLangCache
{
    public class CrossLangCache_MainWorkflow : global::Xap.WorkflowFramework.Workflow
    {


        #region InputFields
        protected Task<global::Platform.Query> Query;
        protected Task<global::Platform.UserIdentification> UserIdentification;
        protected Task<global::Platform.InstrumentationData> InstrumentationData;
        protected Task<global::Platform.Augmentations> Augmentations;
        protected Task<global::Platform.StringData> LegacyCacheEnabled;
        protected Task<global::DeviceCapabilities.DeviceCapabilities_1> aqmDeviceCapabilities;
        protected Task<global::Local.PhonebookRequest_1> pbaKifRequest;
        protected Task<global::Local.Pba.Options_3> pbaOptions;
        #endregion
        #region OutputFields
        protected Task<global::Platform.LegacyQueryResponseData> output;
        #endregion



        public void ExecuteForInheritance()
        {
            this.Execute(
                outputData: out this.output,
                Query: this.Query,
                UserIdentification: this.UserIdentification,
                InstrumentationData: this.InstrumentationData,
                LegacyCacheEnabled: this.LegacyCacheEnabled,
                aqmDeviceCapabilities: this.aqmDeviceCapabilities,
                pbaKifRequest: this.pbaKifRequest,
                pbaOptions: this.pbaOptions
                );
        }

        public virtual void Execute(
            out Task<global::Platform.LegacyQueryResponseData> outputData,
            Task<global::Platform.Query> Query = null,
            Task<global::Platform.UserIdentification> UserIdentification = null,
            Task<global::Platform.InstrumentationData> InstrumentationData = null,
            Task<global::Platform.Augmentations> Augmentations = null,
            Task<global::Platform.StringData> LegacyCacheEnabled = null,
            Task<global::DeviceCapabilities.DeviceCapabilities_1> aqmDeviceCapabilities = null,
            Task<global::Local.PhonebookRequest_1> pbaKifRequest = null,
            Task<global::Local.Pba.Options_3> pbaOptions = null)
        {
            this.BindCrossLangAugmentPlugin();
            this.CrossLangAugmentPlugin.ExecuteForInheritance();

            this.BindCacheQueryPlugin();
            this.CacheQueryPlugin.ExecuteForInheritance();

            this.BindSpellerAnswer();
            this.SpellerAnswer.ExecuteForInheritance();

            this.BindXapQuServiceAnswer();
            this.XapQuServiceAnswer.ExecuteForInheritance();

            

            this.BindCombinedAlterationService();
            this.CombinedAlterationService.ExecuteForInheritance();

            this.BindQRAggregator();
            this.QRAggregator.ExecuteForInheritance();

            this.BindWebAnswer();
            this.WebAnswer.ExecuteForInheritance();

            outputData = this.WebAnswer.output;
        }



        #region SpellerAnswer
        [Timeout("*", 150)]
        public global::SpellerAnswerShell.ISpellerAnswerWorkflow_4_ForInheritance SpellerAnswer { get; } = WorkflowServices.CreateInstance<global::SpellerAnswerShell.ISpellerAnswerWorkflow_4_ForInheritance>();
        protected virtual void BindSpellerAnswer()
        {
            this.SpellerAnswer.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.SpellerAnswer.Inputs.userIdentification = UserIdentification;
            this.SpellerAnswer.Inputs.instrumentationData = InstrumentationData;
            this.SpellerAnswer.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.SpellerAnswer.Inputs.depInputAqrs = null;
            this.SpellerAnswer.Inputs.requiredInputValidationEnabled = null;
            this.SpellerAnswer.Inputs.depRequiredInputAqrs = null;
            this.SpellerAnswer.Inputs.classifierResponses = null;
            this.SpellerAnswer.Inputs.epResponses = null;
            this.SpellerAnswer.Inputs.dependents = null;
            this.SpellerAnswer.Inputs.serviceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"SpellerAnswer"; });
            this.SpellerAnswer.Inputs.virtualServiceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"SpellerAnswer"; });
            this.SpellerAnswer.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });
            this.SpellerAnswer.Inputs.augmentsAndVariants = this.CrossLangAugmentPlugin.modifiedUserAugmentations;
            this.SpellerAnswer.Inputs.customCounterDataEnabled = null;
            this.SpellerAnswer.Inputs.incrementSequenceId = null;
            this.SpellerAnswer.Inputs.languageConfidence = null;
        }
        #endregion SpellerAnswer


        #region CacheQueryPlugin
        [Timeout("*", 300)]
        public ICacheQueryPlugin_ForInheritance CacheQueryPlugin { get;  } = WorkflowServices.CreateInstance<ICacheQueryPlugin_ForInheritance>();
        protected virtual void BindCacheQueryPlugin()
        {
            this.CacheQueryPlugin.Inputs.baseQuery = this.CrossLangAugmentPlugin.modifiedQuery;
            this.CacheQueryPlugin.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.CacheQueryPlugin.Inputs.targetMKT = this.CrossLangAugmentPlugin.targetMKT;
        }
        #endregion CacheQueryPlugin



        #region CombinedAlterationService
        [Timeout("*", 230)]
        public virtual global::CALAnswerShell.ICALAnswerWorkflow_2_ForInheritance CombinedAlterationService { get; } = WorkflowServices.CreateInstance<global::CALAnswerShell.ICALAnswerWorkflow_2_ForInheritance>();
        protected virtual void BindCombinedAlterationService()
        {
            this.CombinedAlterationService.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.CombinedAlterationService.Inputs.dependents = new Aggregate<global::Platform.StringData> { WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"QueryStatsService"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"KoreaAlterationService"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"CommerceQueryTranslator"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"NewsClassifier"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SoccerAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"PhonebookAnswerV2"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"WebAnswer"; }) };
            this.CombinedAlterationService.Inputs.depInputAqrs = new Aggregate<global::Platform.LegacyQueryResponseData> { this.SpellerAnswer.output, this.XapQuServiceAnswer.output };
            this.CombinedAlterationService.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });
            this.CombinedAlterationService.Inputs.instrumentationData = this.InstrumentationData;
            this.CombinedAlterationService.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.CombinedAlterationService.Inputs.serviceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"CombinedAlterationService"; });
            this.CombinedAlterationService.Inputs.userIdentification = this.UserIdentification;
            this.CombinedAlterationService.Inputs.virtualServiceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"CombinedAlterationService"; });
        }
        #endregion CombinedAlterationService



        #region XapQuServiceAnswer
        [Timeout("*", 80)]
        public virtual global::QAS.IMainWorkflow_1_ForInheritance XapQuServiceAnswer { get; } = WorkflowServices.CreateInstance<global::QAS.IMainWorkflow_1_ForInheritance>();
        protected virtual void BindXapQuServiceAnswer()
        {
            this.XapQuServiceAnswer.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.XapQuServiceAnswer.Inputs.instrumentationData = this.InstrumentationData;
            this.XapQuServiceAnswer.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.XapQuServiceAnswer.Inputs.userIdentification = this.UserIdentification;
            this.XapQuServiceAnswer.Inputs.virtualService = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"XapQuServiceAnswer"; });
        }
        #endregion XapQuServiceAnswer

        #region QRAggregator
        [Timeout("*", 350)]
        public virtual global::QRAggregator.IAlterationSelectorWorkflow_1_ForInheritance QRAggregator { get; } = WorkflowServices.CreateInstance<global::QRAggregator.IAlterationSelectorWorkflow_1_ForInheritance>();
        protected virtual void BindQRAggregator()
        {
            this.QRAggregator.Inputs.augmentations = this.Augmentations;
            this.QRAggregator.Inputs.calResponse = this.CombinedAlterationService.output_bond;
            this.QRAggregator.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.QRAggregator.Inputs.spellerResponse = this.SpellerAnswer.output_QU_AnnotationProviderResponse_1;
        }
        #endregion


        #region WebAnswer
        [Timeout("*", 5000)]
        public virtual global::Platform.IWebAnswerComposer_3_ForInheritance WebAnswer { get; } = WorkflowServices.CreateInstance<global::Platform.IWebAnswerComposer_3_ForInheritance>();
        protected virtual void BindWebAnswer()
        {
            this.WebAnswer.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.WebAnswer.Inputs.dependents = new Aggregate<global::Platform.StringData> { WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"QueryWordBagGen"; }) };
            this.WebAnswer.Inputs.depInputAqrs = new Aggregate<global::Platform.LegacyQueryResponseData> { this.CombinedAlterationService.output,this.QRAggregator.alterationOutputAQR, this.SpellerAnswer.output, this.XapQuServiceAnswer.output };
            this.WebAnswer.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });
            this.WebAnswer.Inputs.instrumentationData = this.InstrumentationData;
            this.WebAnswer.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.WebAnswer.Inputs.augmentsAndVariants = this.CrossLangAugmentPlugin.modifiedUserAugmentations;
            this.WebAnswer.Inputs.serviceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"WebAnswer"; });
            this.WebAnswer.Inputs.userIdentification = this.UserIdentification;
            this.WebAnswer.Inputs.virtualServiceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"WebAnswer"; });
        }
        #endregion WebAnswer


        #region CrossLangAugmentationsPlugin
        [Timeout("*", 350)]
        public ICrossLangAugmentPlugin_ForInheritance CrossLangAugmentPlugin { get; } = WorkflowServices.CreateInstance<ICrossLangAugmentPlugin_ForInheritance>();
        protected virtual void BindCrossLangAugmentPlugin()
        {
            this.CrossLangAugmentPlugin.Inputs.query = this.Query;
            this.CrossLangAugmentPlugin.Inputs.augmentations = this.Augmentations;
        }
        #endregion


    }
}
