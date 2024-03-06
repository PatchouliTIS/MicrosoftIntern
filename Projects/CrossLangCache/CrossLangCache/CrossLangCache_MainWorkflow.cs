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
                Augmentations: this.Augmentations,
                LocationContext: this.LocationContext,
                cuRequestData: this.cuRequestData);
        }

        public void Execute(
            out Task<global::Platform.LegacyQueryResponseData> outputData,
            Task<global::Platform.Query> Query = null,
            Task<global::Platform.UserIdentification> UserIdentification = null,
            Task<global::Platform.InstrumentationData> InstrumentationData = null,
            Task<global::Platform.Augmentations> Augmentations = null,
            Task<global::Local.LocationContext_1> LocationContext = null,
            Task<global::Cortana.CU.RequestData> cuRequestData = null)
        {
            this.Query = Query;
            this.UserIdentification = UserIdentification;
            this.InstrumentationData = InstrumentationData;
            this.Augmentations = Augmentations;
            this.LocationContext = LocationContext;
            this.cuRequestData = cuRequestData;


            this.BindReverseGeocoder();
            this.ReverseGeocoder.ExecuteForInheritance();

            this.BindCrossLangAugmentPlugin();
            this.CrossLangAugmentPlugin.ExecuteForInheritance();

            this.BindSessionReader();
            this.SessionReader.ExecuteForInheritance();


            this.BindCqnaPreweb();
            this.CqnaPreweb.ExecuteForInheritance();


            this.BindCacheQueryPlugin();
            this.CacheQueryPlugin.ExecuteForInheritance();

            this.BindLocationShift();
            this.LocationShift.ExecuteForInheritance();

            this.BingQueryMTServiceInputs();
            this.BingQueryMTWorkflow.ExecuteForInheritance();


            this.BindLanguagePrediction();
            this.QueryLanguage.ExecuteForInheritance();


            this.BindCALMultiQueryGeneratorCondition();
            AsyncIf(this.WBMultiQueryWorkflowCondition.ExecuteForInheritance()).Then(() =>
            {
                this.BindCALMultiQueryGeneratorWorkflow();
                this.WBMultiQueryWorkflow.ExecuteForInheritance();
            });



            this.BindNewlyNavObjectStoreLookupMQWorkflow();
            this.NewlyNavObjectStoreLookupMQWorkflow.ExecuteForInheritance();

            this.BindXapQuServiceAnswer();
            this.XapQuServiceAnswer.ExecuteForInheritance();

            this.BindXapQuAthena();
            this.XapQuAthena.ExecuteForInheritance();

            this.BindXapQuWithOMA();
            this.XapQuWithOMA.ExecuteForInheritance();

            this.BindHybridLes();
            this.HybridLes.ExecuteForInheritance();

            this.BindLangDetector();
            this.LangDetector.ExecuteForInheritance();

            this.BindQRNeutralQueryClassifier();
            this.QRNeutralQueryClassifier.ExecuteForInheritance();

            this.BindConvertQueryLanguage();
            this.ConvertQueryLanguage.ExecuteForInheritance();




            this.BindSpellerAnswer();
            this.SpellerAnswer.ExecuteForInheritance();


            this.BindSpellerAnswerSecond();
            this.SpellerAnswerSecond.ExecuteForInheritance();

            this.BindSpellerCombiner();
            this.SpellerCombiner.ExecuteForInheritance();


            this.BindSpellerProcessorPlugin();
            this.SpellerProcessorPlugin.ExecuteForInheritance();

            this.BindPreQueryParserConversationEngine();
            this.PreQueryParserConversationEngine.ExecuteForInheritance();

            this.BindQueryParserV3();
            this.QueryParserV3.ExecuteForInheritance();


            this.BindDolphinAdapterWorkflow();
            this.DolphinAdapterWorkflow.ExecuteForInheritance();


            this.BindQASWithSpeller();
            this.QASWithSpeller.ExecuteForInheritance();

            this.BindContextualQRSelectorWithSpeller();
            this.ContextualQRSelectorWithSpeller.ExecuteForInheritance();

            this.BindBuildCalOutput();
            this.BuildCalOutput.ExecuteForInheritance();


            this.BindCALMultiQuery();
            this.CALMultiQuery.ExecuteForInheritance();


            this.BindCombinedAlterationService();
            this.CombinedAlterationService.ExecuteForInheritance();

            this.BindisNoResultsEnabled();
            this.isNoResultsEnabled.ExecuteForInheritance();

            this.BindQRAggregator();
            this.QRAggregator.ExecuteForInheritance();



            this.BindQASMigrationWorkflow();
            this.QASMigrationWorkflow.ExecuteForInheritance();


            this.BindContextualQLFMain();
            this.ContextualQLFMain.ExecuteForInheritance();

            this.BindFlightAssignmentServiceWithSegmentRelevanceLBFA();
            this.FlightAssignmentServiceWithSegmentRelevanceLBFA.ExecuteForInheritance();

            this.BindSpellerFLE();
            this.SpellerFLE.ExecuteForInheritance();

            this.BindWebAnswer();
            this.WebAnswer.ExecuteForInheritance();


            this.BindORCAQueryLanguageCondition();
            AsyncIf(this.ORCAQueryLanguageCondition.ExecuteForInheritance())
            .Then(() =>
            {
                this.BindORCAQueryLanguagePassthrough();
                ORCAQueryLanguagePassthrough.ExecuteForInheritance();
            });

            this.BindORCADeepIntentWorkflow();
            ORCADeepIntentWorkflow.ExecuteForInheritance();



            this.BindQueryIndex();
            this.QueryIndex.ExecuteForInheritance();



            //this.BindAQRCombiner();
            //this.AQRCombiner.ExecuteForInheritance();

            this.BindAnswerLoggerPlugin();
            this.AnswerLoggerPlugin.ExecuteForInheritance();


            outputData = this.AnswerLoggerPlugin.output;
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
            this.CacheQueryPlugin.Inputs.baseQuery = this.CrossLangAugmentPlugin.modifiedQuery;
            this.CacheQueryPlugin.Inputs.targetMKT = this.CrossLangAugmentPlugin.targetMKT;
        }
        #endregion CacheQueryPlugin


        #region LangDetector
        public global::QR.LanguageDetector.Workflows.ILanguageDetectorWorkflow_1_ForInheritance LangDetector { get; } = WorkflowServices.CreateInstance<global::QR.LanguageDetector.Workflows.ILanguageDetectorWorkflow_1_ForInheritance>();
        protected void BindLangDetector()
        {
            this.LangDetector.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion

        #region QRNeutralQueryClassifier
        public virtual global::QR.NeutralQueryClassifier.Plugins.IClassifierPlugin_ForInheritance QRNeutralQueryClassifier { get; } = WorkflowServices.CreateInstance<global::QR.NeutralQueryClassifier.Plugins.IClassifierPlugin_ForInheritance>();
        protected virtual void BindQRNeutralQueryClassifier()
        {
            this.QRNeutralQueryClassifier.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion


        #region ConvertQueryLanguage
        public virtual global::QR.ConvertQueryLanguage.Plugins.IConvertQueryLanguagePlugin_ForInheritance ConvertQueryLanguage { get; } = WorkflowServices.CreateInstance<global::QR.ConvertQueryLanguage.Plugins.IConvertQueryLanguagePlugin_ForInheritance>();
        protected virtual void BindConvertQueryLanguage()
        {
            this.ConvertQueryLanguage.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.ConvertQueryLanguage.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
        }
        #endregion ConvertQueryLanguage



        #region ReverseGeocoder
        public virtual global::Local.Workflows.IReverseGeocoder_1_ForInheritance ReverseGeocoder { get; } = WorkflowServices.CreateInstance<global::Local.Workflows.IReverseGeocoder_1_ForInheritance>();
        protected virtual void BindReverseGeocoder()
        {
            this.ReverseGeocoder.Inputs.locationContext = this.LocationContext;
        }
        #endregion ReverseGeocoder


        #region SpellerAnswer
        [Timeout("*", 150)]
        public global::SpellerAnswerShell.ISpellerAnswerWorkflow_4_ForInheritance SpellerAnswer { get; } = WorkflowServices.CreateInstance<global::SpellerAnswerShell.ISpellerAnswerWorkflow_4_ForInheritance>();
        protected virtual void BindSpellerAnswer()
        {
            this.SpellerAnswer.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.SpellerAnswer.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            // this.SpellerAnswer.Inputs.augmentsAndVariants = this.CrossLangAugmentPlugin.modifiedUserAugmentations;
            this.SpellerAnswer.Inputs.augmentsAndVariants = WorkflowServices.CreateDataSource<global::Platform.LegacyPluginAugmentsAndVariantsData>(t => { t.Augmentation = @"[JointOptimization IsJointOptimizationKifOutputEnabled=""1""]"; });
            this.SpellerAnswer.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });
            this.SpellerAnswer.Inputs.instrumentationData = this.InstrumentationData;
            this.SpellerAnswer.Inputs.serviceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"SpellerAnswer"; });
            this.SpellerAnswer.Inputs.userIdentification = this.UserIdentification;
            this.SpellerAnswer.Inputs.virtualServiceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"SpellerAnswer"; });
            this.SpellerAnswer.Inputs.languageConfidence = this.LangDetector.firstLanguage;
            this.SpellerAnswer.Inputs.qrContexts = new Aggregate<global::QR.Context.Context>();
            this.SpellerAnswer.Inputs.qrContexts.Add(this.ConvertQueryLanguage.output);
            this.SpellerAnswer.Inputs.qrContexts.Add(this.QRNeutralQueryClassifier.output);
        }
        #endregion SpellerAnswer

        #region SpellerAnswerSecond
        public global::SpellerAnswerShell.ISpellerAnswerWorkflow_4_ForInheritance SpellerAnswerSecond { get; } = WorkflowServices.CreateInstance<global::SpellerAnswerShell.ISpellerAnswerWorkflow_4_ForInheritance>();
        protected void BindSpellerAnswerSecond()
        {
            this.SpellerAnswerSecond.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.SpellerAnswerSecond.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });
            this.SpellerAnswerSecond.Inputs.instrumentationData = this.InstrumentationData;
            this.SpellerAnswerSecond.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.SpellerAnswerSecond.Inputs.serviceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"SpellerAnswer"; });
            this.SpellerAnswerSecond.Inputs.userIdentification = this.UserIdentification;
            this.SpellerAnswerSecond.Inputs.virtualServiceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"SpellerAnswerSecond"; });
            this.SpellerAnswerSecond.Inputs.augmentsAndVariants = WorkflowServices.CreateDataSource<global::Platform.LegacyPluginAugmentsAndVariantsData>(t => { t.Augmentation = @"[JointOptimization IsJointOptimizationKifOutputEnabled=""1""]"; });
            // this.SpellerAnswerSecond.Inputs.augmentsAndVariants = this.CrossLangAugmentPlugin.modifiedUserAugmentations;
            this.SpellerAnswerSecond.Inputs.languageConfidence = this.LangDetector.secondLanguage;
        }
        #endregion

        #region SpellerCombiner
        public global::QR.SpellerCombiner.Plugins.ISpellerCombiner_ForInheritance SpellerCombiner { get; } = WorkflowServices.CreateInstance<global::QR.SpellerCombiner.Plugins.ISpellerCombiner_ForInheritance>();
        protected void BindSpellerCombiner()
        {
            this.SpellerCombiner.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.SpellerCombiner.Inputs.spellerAnswer1 = this.SpellerAnswer.output;
            this.SpellerCombiner.Inputs.spellerAnswerKif1 = this.SpellerAnswer.output_QU_AnnotationProviderResponse_1;
            this.SpellerCombiner.Inputs.spellerAnswerLanguage1 = this.LangDetector.firstLanguage;
            this.SpellerCombiner.Inputs.spellerAnswerIsOriginalMarket1 = this.SpellerAnswer.isOriginalMarket;
            this.SpellerCombiner.Inputs.spellerAnswer2 = this.SpellerAnswerSecond.output;
            this.SpellerCombiner.Inputs.spellerAnswerKif2 = this.SpellerAnswerSecond.output_QU_AnnotationProviderResponse_1;
            this.SpellerCombiner.Inputs.spellerAnswerLanguage2 = this.LangDetector.secondLanguage;
            this.SpellerCombiner.Inputs.spellerAnswerIsOriginalMarket2 = this.SpellerAnswerSecond.isOriginalMarket;
        }
        #endregion


        #region locationRanker
        public virtual global::Blis.Main.IRead_7_ForInheritance locationRanker { get; } = WorkflowServices.CreateInstance<global::Blis.Main.IRead_7_ForInheritance>();
        protected virtual void BindlocationRanker()
        {
            this.locationRanker.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.locationRanker.Inputs.defaultLocationContext = this.LocationContext;
            this.locationRanker.Inputs.guidGeoChain = this.ReverseGeocoder.guidGeoChain;
            this.locationRanker.Inputs.instrumentationData = this.InstrumentationData;
            this.locationRanker.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.locationRanker.Inputs.reverseGeocoderContext = this.ReverseGeocoder.outputLocationContext;
            this.locationRanker.Inputs.userIdentification = this.UserIdentification;
        }
        #endregion locationRanker

        #region XapQuWithOMA
        [Timeout("*", 220)]
        public virtual global::QAS.IMainWorkflow_2_ForInheritance XapQuWithOMA { get; } = WorkflowServices.CreateInstance<global::QAS.IMainWorkflow_2_ForInheritance>();
        protected virtual void BindXapQuWithOMA()
        {
            this.XapQuWithOMA.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.XapQuWithOMA.Inputs.instrumentationData = this.InstrumentationData;
            this.XapQuWithOMA.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.XapQuWithOMA.Inputs.userIdentification = this.UserIdentification;
            // TODO
            this.XapQuWithOMA.Inputs.augmentsAndVariants = WorkflowServices.CreateDataSource<global::Platform.LegacyPluginAugmentsAndVariantsData>(t => { t.Augmentation = @"[XAPQuServiceAnswer WebModels=""1""][AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]"; });
            this.XapQuWithOMA.Inputs.input_Local_LocationContext_1 = this.locationRanker.locationContextForQas;
            this.XapQuWithOMA.Inputs.virtualService = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"XapQuWithOma"; });
        }
        #endregion XapQuWithOMA



        #region XapQuServiceAnswer
        [Timeout("*", 80)]
        public virtual global::QAS.IMainWorkflow_1_ForInheritance XapQuServiceAnswer { get; } = WorkflowServices.CreateInstance<global::QAS.IMainWorkflow_1_ForInheritance>();
        protected virtual void BindXapQuServiceAnswer()
        {
            this.XapQuServiceAnswer.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.XapQuServiceAnswer.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.XapQuServiceAnswer.Inputs.instrumentationData = this.InstrumentationData;
            this.XapQuServiceAnswer.Inputs.userIdentification = this.UserIdentification;
            this.XapQuServiceAnswer.Inputs.virtualService = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"XapQuServiceAnswer"; });
            this.XapQuServiceAnswer.Inputs.augmentsAndVariants = this.CrossLangAugmentPlugin.modifiedUserAugmentations;
            this.XapQuServiceAnswer.Inputs.dependents = new Aggregate<global::Platform.StringData> { WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SportsTennisAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SportsAFLAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SportsAutoRacesAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SportsCBKAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SportsCricketAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SportsEntityAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SportsGolfAnswer"; }) };
            this.XapQuServiceAnswer.Inputs.input_Local_LocationContext_1 = this.ReverseGeocoder.outputLocationContext;
            this.XapQuServiceAnswer.Inputs.virtualService = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"XapQuServiceAnswer"; });
        }
        #endregion XapQuServiceAnswer



        #region BuildCalOutput
        [Timeout("*", 350)]
        public virtual global::ContextualAnswers.IBuildCalOutput_1_ForInheritance BuildCalOutput { get; } = WorkflowServices.CreateInstance<global::ContextualAnswers.IBuildCalOutput_1_ForInheritance>();
        protected virtual void BindBuildCalOutput()
        {
            this.BuildCalOutput.Inputs.queryRepresentationResponse = this.QASWithSpeller.contextualQueryRepresentationResponse;
        }
        #endregion BuildCalOutput



        #region ClientDateTimeExtractor
        public virtual global::Answer.Shared.IClientDateTimeExtractor_1_ForInheritance ClientDateTimeExtractor { get; } = WorkflowServices.CreateInstance<global::Answer.Shared.IClientDateTimeExtractor_1_ForInheritance>();
        protected virtual void BindClientDateTimeExtractor()
        {
            this.ClientDateTimeExtractor.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
        }
        #endregion ClientDateTimeExtractor



        #region LocalMultiTurnPreQas
        public virtual global::Local.IMultiTurnPreQas_2_ForInheritance LocalMultiTurnPreQas { get; } = WorkflowServices.CreateInstance<global::Local.IMultiTurnPreQas_2_ForInheritance>();
        protected virtual void BindLocalMultiTurnPreQas()
        {
            this.LocalMultiTurnPreQas.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.LocalMultiTurnPreQas.Inputs.cuRequestData = this.cuRequestData;
            this.LocalMultiTurnPreQas.Inputs.session = this.SessionReader.session;
        }
        #endregion LocalMultiTurnPreQas

        #region QasBufferCreator
        public virtual global::QAS.IBufferCreator_1_ForInheritance QasBufferCreator { get; } = WorkflowServices.CreateInstance<global::QAS.IBufferCreator_1_ForInheritance>();
        protected virtual void BindQasBufferCreator()
        {
            this.QasBufferCreator.Inputs.featurizationOutputs = new Aggregate<global::QAS.Inmemory.QueryRepresentation.FeaturizationOutput> {
                this.ClientDateTimeExtractor.clientDateTimeQasFeature,
                this.LocalMultiTurnPreQas.qasFeatures
            };
        }
        #endregion QasBufferCreator

        #region QASWithSpeller
        [Timeout("*", 120)]
        public virtual global::QAS.IMainWorkflow_2_ForInheritance QASWithSpeller { get; } = WorkflowServices.CreateInstance<global::QAS.IMainWorkflow_2_ForInheritance>();
        protected virtual void BindQASWithSpeller()
        {
            this.QASWithSpeller.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.QASWithSpeller.Inputs.depInputAqrs = new Aggregate<global::Platform.LegacyQueryResponseData> { this.SpellerAnswer.output, this.XapQuServiceAnswer.output, this.XapQuWithOMA.output };
            this.QASWithSpeller.Inputs.instrumentationData = this.InstrumentationData;
            this.QASWithSpeller.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.QASWithSpeller.Inputs.userIdentification = this.UserIdentification;
            this.QASWithSpeller.Inputs.virtualService = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"QASWithSpeller"; });
            this.QASWithSpeller.Inputs.augmentsAndVariants = WorkflowServices.CreateDataSource<global::Platform.LegacyPluginAugmentsAndVariantsData>(t => { t.Augmentation = @"[XAPQuServiceAnswer VirtualService=""QASWithSpeller"" DefaultQueryViewName=""SpellCorrectedQuery1""][AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]"; });
            this.QASWithSpeller.Inputs.defaultQueryView = WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SpellCorrectedQuery1"; });
            this.QASWithSpeller.Inputs.input_Local_LocationContext_1 = this.locationRanker.locationContextForQas;
            this.QASWithSpeller.Inputs.inputBuffers = this.QasBufferCreator.qasBuffer;
            this.QASWithSpeller.Inputs.session = this.SessionReader.session;
            this.QASWithSpeller.Inputs.virtualService = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"QASWithSpeller"; });
            this.QASWithSpeller.Inputs.dependents = (this.QASWithSpeller.Inputs.dependents as Aggregate<global::Platform.StringData>) ?? new Aggregate<global::Platform.StringData>();
            this.QASWithSpeller.Inputs.dependents.Add(WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"TopicalAuthoritiesAnswer"; }));
            this.QASWithSpeller.Inputs.dependents.Add(WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"AdService"; }));
            this.QASWithSpeller.Inputs.dependents.Add(WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"MsnJVDataAnswer"; }));
            this.QASWithSpeller.Inputs.dependents.Add(WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"PhonebookAnswerV2"; }));
            this.QASWithSpeller.Inputs.dependents.Add(WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"TranslateThisAnswer"; }));
        }
        #endregion QASWithSpeller



        #region SessionReader
        public virtual global::SessionManager.ISessionReader_4_ForInheritance SessionReader { get; } = WorkflowServices.CreateInstance<global::SessionManager.ISessionReader_4_ForInheritance>();
        protected virtual void BindSessionReader()
        {
            this.SessionReader.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.SessionReader.Inputs.instrumentationData = this.InstrumentationData;
        }
        #endregion SessionReader

        #region CqnaPreweb
        public virtual global::EntityIdLookUpAnswer.Cqna.IPrewebCqnaWorkflow_103_ForInheritance CqnaPreweb { get; } = WorkflowServices.CreateInstance<global::EntityIdLookUpAnswer.Cqna.IPrewebCqnaWorkflow_103_ForInheritance>();
        protected virtual void BindCqnaPreweb()
        {
            this.CqnaPreweb.Inputs.Augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.CqnaPreweb.Inputs.GoostmanSession = this.SessionReader.session;
            this.CqnaPreweb.Inputs.GoostmanSessionId = this.SessionReader.sessionId;
            this.CqnaPreweb.Inputs.Query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion CqnaPreweb


        #region NewlyNavObjectStoreLookupMQWorkflow
        public virtual global::FreshNewlyNav.INewlyNavObjectStoreLookupMQWorkflow_2_ForInheritance NewlyNavObjectStoreLookupMQWorkflow { get; } = WorkflowServices.CreateInstance<global::FreshNewlyNav.INewlyNavObjectStoreLookupMQWorkflow_2_ForInheritance>();
        protected virtual void BindNewlyNavObjectStoreLookupMQWorkflow()
        {
            this.NewlyNavObjectStoreLookupMQWorkflow.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion NewlyNavObjectStoreLookupMQWorkflow

        #region CALMultiQuery
        [Timeout("*", 230)]
        public virtual global::CALPlugins.ICALMQInputWithQASWorkflow_2_ForInheritance CALMultiQuery { get; } = WorkflowServices.CreateInstance<global::CALPlugins.ICALMQInputWithQASWorkflow_2_ForInheritance>();
        protected virtual void BindCALMultiQuery()
        {
            this.CALMultiQuery.Inputs.inputPathSets = new Aggregate<global::MultiQuery.PathSet_2> { this.BuildCalOutput.pathSet_2, this.CqnaPreweb.outputForCal, this.CqnaPreweb.outputPronounBasedForCal, this.CqnaPreweb.outputPronounBasedFromDLIS, this.NewlyNavObjectStoreLookupMQWorkflow.MultiQueryPathSet };
            this.CALMultiQuery.Inputs.qasBond = this.ContextualQRSelectorWithSpeller.output;
            this.CALMultiQuery.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion CALMultiQuery

        #region ContextualQRSelectorWithSpeller
        [Timeout("*", 350)]
        public virtual global::ContextualAnswers.IContextualQRSelector_1_ForInheritance ContextualQRSelectorWithSpeller { get; } = WorkflowServices.CreateInstance<global::ContextualAnswers.IContextualQRSelector_1_ForInheritance>();
        protected virtual void BindContextualQRSelectorWithSpeller()
        {
            this.ContextualQRSelectorWithSpeller.Inputs.contextualQueryRepresentationResponse = this.QASWithSpeller.contextualQueryRepresentationResponse;
            this.ContextualQRSelectorWithSpeller.Inputs.queryRepresentationResponse = this.QASWithSpeller.queryRepresentationResponse;
        }
        #endregion ContextualQRSelectorWithSpeller


        #region CombinedAlterationService
        [Timeout("*", 230)]
        public virtual global::CALAnswerShell.ICALAnswerWorkflow_2_ForInheritance CombinedAlterationService { get; } = WorkflowServices.CreateInstance<global::CALAnswerShell.ICALAnswerWorkflow_2_ForInheritance>();
        protected virtual void BindCombinedAlterationService()
        {
            this.CombinedAlterationService.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.CombinedAlterationService.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.CombinedAlterationService.Inputs.dependents = new Aggregate<global::Platform.StringData> { WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"QueryStatsService"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"KoreaAlterationService"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"CommerceQueryTranslator"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"NewsClassifier"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"SoccerAnswer"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"PhonebookAnswerV2"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"WebAnswer"; }) };
            this.CombinedAlterationService.Inputs.depInputAqrs = new Aggregate<global::Platform.LegacyQueryResponseData> { this.SpellerAnswer.output, this.XapQuServiceAnswer.output };
            this.CombinedAlterationService.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });
            this.CombinedAlterationService.Inputs.instrumentationData = this.InstrumentationData;
            this.CombinedAlterationService.Inputs.serviceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"CombinedAlterationService"; });
            this.CombinedAlterationService.Inputs.userIdentification = this.UserIdentification;
            this.CombinedAlterationService.Inputs.virtualServiceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"CombinedAlterationService"; });
            // this.CombinedAlterationService.Inputs.augmentsAndVariants = WorkflowServices.CreateDataSource<global::Platform.LegacyPluginAugmentsAndVariantsData>(t => { t.Augmentation = @"[AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]"; });
            this.CombinedAlterationService.Inputs.augmentsAndVariants = this.CrossLangAugmentPlugin.modifiedUserAugmentations;
            this.CombinedAlterationService.Inputs.depInputAqrs = new Aggregate<global::Platform.LegacyQueryResponseData>();
            this.CombinedAlterationService.Inputs.depInputAqrs.Add(this.CALMultiQuery.output);
            // this.CombinedAlterationService.Inputs.depInputAqrs.Remove(this.SpellerAnswer.output);
            this.CombinedAlterationService.Inputs.depInputAqrs.Add(this.SpellerCombiner.spellerAnswerCombined);
            this.CombinedAlterationService.Inputs.depInputAqrs.Add(this.WBMultiQueryWorkflow.pathSetOutput);
            this.CombinedAlterationService.Inputs.qrContexts = new Aggregate<global::QR.Context.Context>();
            this.CombinedAlterationService.Inputs.qrContexts.Add(this.ConvertQueryLanguage.output);
            this.CombinedAlterationService.Inputs.languageConfidence = this.SpellerCombiner.spellerAnswerCombinedLanguage;

        }
        #endregion CombinedAlterationService


        #region CALMultiQueryGeneratorWorkflow
        public virtual global::WordBreakerMultiplePath.Plugins.ICALMultiQueryWBCondition_ForInheritance WBMultiQueryWorkflowCondition { get; } = WorkflowServices.CreateInstance<global::WordBreakerMultiplePath.Plugins.ICALMultiQueryWBCondition_ForInheritance>();
        protected virtual void BindCALMultiQueryGeneratorCondition()
        {
            this.WBMultiQueryWorkflowCondition.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        public virtual global::WordBreakerMultiplePath.ICALMultiQueryGeneratorWorkflow_ForInheritance WBMultiQueryWorkflow { get; } = WorkflowServices.CreateInstance<global::WordBreakerMultiplePath.ICALMultiQueryGeneratorWorkflow_ForInheritance>();
        protected virtual void BindCALMultiQueryGeneratorWorkflow()
        {
            this.WBMultiQueryWorkflow.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion CALMultiQueryGeneratorWorkflow


        #region QRAggregator
        [Timeout("*", 350)]
        public virtual global::QRAggregator.IAlterationSelectorWorkflow_1_ForInheritance QRAggregator { get; } = WorkflowServices.CreateInstance<global::QRAggregator.IAlterationSelectorWorkflow_1_ForInheritance>();
        protected virtual void BindQRAggregator()
        {
            this.QRAggregator.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.QRAggregator.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.QRAggregator.Inputs.calResponse = this.CombinedAlterationService.output_bond;
            this.QRAggregator.Inputs.spellerResponse = this.SpellerAnswer.output_QU_AnnotationProviderResponse_1;
        }
        #endregion


        #region isNoResultsEnabled
        [Timeout("*", 50)]
        public virtual global::IsNoResultsEnabled.IEvaluate_3_ForInheritance isNoResultsEnabled { get; } = WorkflowServices.CreateInstance<global::IsNoResultsEnabled.IEvaluate_3_ForInheritance>();
        protected virtual void BindisNoResultsEnabled()
        {
            this.isNoResultsEnabled.Inputs.augmentation = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.isNoResultsEnabled.Inputs.caloutput = this.CombinedAlterationService.output_bond;
            this.isNoResultsEnabled.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion isNoResultsEnabled


        #region QASMigrationWorkflow

        public virtual QAS.IMigrationWorkflow_1_ForInheritance QASMigrationWorkflow { get; } = WorkflowServices.CreateInstance<QAS.IMigrationWorkflow_1_ForInheritance>();

        protected virtual void BindQASMigrationWorkflow()
        {
            this.QASMigrationWorkflow.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.QASMigrationWorkflow.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion


        #region ContextualQLFMain
        [Timeout("*", 2000)]
        public virtual global::ContextualQLF.IMain_1_ForInheritance ContextualQLFMain { get; } = WorkflowServices.CreateInstance<global::ContextualQLF.IMain_1_ForInheritance>();
        protected virtual void BindContextualQLFMain()
        {
            this.ContextualQLFMain.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.ContextualQLFMain.Inputs.locationContext = this.LocationContext;
            this.ContextualQLFMain.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion ContextualQLFMain

        #region FlightAssignmentServiceWithSegmentRelevanceLBFA
        [Timeout("*", 100)]
        public virtual global::LateBoundFlightAssignment.ISegmentRelevanceWorkflow_4_ForInheritance FlightAssignmentServiceWithSegmentRelevanceLBFA { get; } = WorkflowServices.CreateInstance<global::LateBoundFlightAssignment.ISegmentRelevanceWorkflow_4_ForInheritance>();
        protected virtual void BindFlightAssignmentServiceWithSegmentRelevanceLBFA()
        {
            this.FlightAssignmentServiceWithSegmentRelevanceLBFA.Inputs.Augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.FlightAssignmentServiceWithSegmentRelevanceLBFA.Inputs.InstrumentationData = this.InstrumentationData;
            this.FlightAssignmentServiceWithSegmentRelevanceLBFA.Inputs.QueryProcessingItems = this.XapQuServiceAnswer.output_QAS_QPI_1;
        }
        #endregion FlightAssignmentServiceWithSegmentRelevanceLBFA

        #region XapQuAthena
        [Timeout("*", 220)]
        public virtual global::QAS.IMainWorkflow_2_ForInheritance XapQuAthena { get; } = WorkflowServices.CreateInstance<global::QAS.IMainWorkflow_2_ForInheritance>();
        protected virtual void BindXapQuAthena()
        {
            this.XapQuAthena.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.XapQuAthena.Inputs.augmentsAndVariants = WorkflowServices.CreateDataSource<global::Platform.LegacyPluginAugmentsAndVariantsData>(t => { t.Augmentation = @"[XAPQuServiceAnswer VirtualService=""Athena""]"; });
            this.XapQuAthena.Inputs.dependents = new Aggregate<global::Platform.StringData> { WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"MMQueryClassifierService"; }), WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"MultimediaKifVideoAnswer"; }) };
            this.XapQuAthena.Inputs.instrumentationData = this.InstrumentationData;
            this.XapQuAthena.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.XapQuAthena.Inputs.userIdentification = this.UserIdentification;
            this.XapQuAthena.Inputs.virtualService = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"Athena"; });
        }
        #endregion XapQuAthena


        #region PreQueryParserConversationEngine
        public virtual global::Dolphin.Workflows.IPreQueryParserConversationWorkflow_45_ForInheritance PreQueryParserConversationEngine { get; } = WorkflowServices.CreateInstance<global::Dolphin.Workflows.IPreQueryParserConversationWorkflow_45_ForInheritance>();
        protected virtual void BindPreQueryParserConversationEngine()
        {
            this.PreQueryParserConversationEngine.Inputs.Augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.PreQueryParserConversationEngine.Inputs.Query = this.CacheQueryPlugin.outputQuery;
            this.PreQueryParserConversationEngine.Inputs.SessionHistory = this.SessionReader.session;
            this.PreQueryParserConversationEngine.Inputs.SpellerOutput = this.SpellerProcessorPlugin.spellerOutput;
        }
        #endregion PreQueryParserConversationEngine

        #region BingQueryMTService
        public virtual global::BingQueryMTService.Workflows.IBingQueryMTWorkflow_1_ForInheritance BingQueryMTWorkflow { get; } = WorkflowServices.CreateInstance<global::BingQueryMTService.Workflows.IBingQueryMTWorkflow_1_ForInheritance>();
        protected virtual void BingQueryMTServiceInputs()
        {
            this.BingQueryMTWorkflow.Inputs.query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion

        #region QueryParserV3
        [Timeout("*", 350)]
        public virtual global::Dolphin.Workflows.IQueryParserV3_46_ForInheritance QueryParserV3 { get; } = WorkflowServices.CreateInstance<global::Dolphin.Workflows.IQueryParserV3_46_ForInheritance>();
        protected virtual void BindQueryParserV3()
        {
            this.QueryParserV3.Inputs.AthenaQasResponse = this.XapQuAthena.queryRepresentationResponse;
            this.QueryParserV3.Inputs.Augmentations = this.Augmentations;
            this.QueryParserV3.Inputs.ConversationWorkflowContext = this.PreQueryParserConversationEngine.outputWorkflowContext;
            this.QueryParserV3.Inputs.InstrumentationData = this.InstrumentationData;
            this.QueryParserV3.Inputs.Query = this.Query;
            this.QueryParserV3.Inputs.SpellerAnswer = this.SpellerAnswer.output;
            this.QueryParserV3.Inputs.SpellerOutput = this.SpellerProcessorPlugin.spellerOutput;
            this.QueryParserV3.Inputs.UserIdentification = this.UserIdentification;
            this.QueryParserV3.Inputs.BingQueryMTResult = this.BingQueryMTWorkflow.result;
        }
        #endregion QueryParserV3

        #region QueryIndex
        public virtual global::QueryIndex.IMainWorkflow_16_ForInheritance QueryIndex { get; } = WorkflowServices.CreateInstance<global::QueryIndex.IMainWorkflow_16_ForInheritance>();
        protected virtual void BindQueryIndex()
        {
            this.QueryIndex.Inputs.alterationOutput = this.QRAggregator.alterationOutput;
            this.QueryIndex.Inputs.augmentations = this.Augmentations;
            this.QueryIndex.Inputs.dolphinTokenResult = this.QueryParserV3.TokenResult;
            this.QueryIndex.Inputs.inputParserResult = this.QueryParserV3.resultV3;
            this.QueryIndex.Inputs.inputQasResponse = this.XapQuServiceAnswer.queryRepresentationResponse;
            this.QueryIndex.Inputs.qasAnswerTypes = this.QueryParserV3.QasAnswerType;
            this.QueryIndex.Inputs.qasEqnaIntentScores = this.QueryParserV3.QasEqnaIntentScores;
            this.QueryIndex.Inputs.query = this.Query;
            this.QueryIndex.Inputs.spellerOutput = this.SpellerProcessorPlugin.spellerOutput;
            this.QueryIndex.Inputs.webResponse = this.WebAnswer.webResponse;
            this.QueryIndex.Inputs.webResults = this.WebAnswer.webResults;
        }
        #endregion QueryIndex

        #region LanguagePrediction
        [Timeout("*", 220)]
        public virtual global::LanguagePrediction.IOSDLISWorkflow_ForInheritance QueryLanguage { get; } = WorkflowServices.CreateInstance<global::LanguagePrediction.IOSDLISWorkflow_ForInheritance>();
        protected virtual void BindLanguagePrediction()
        {
            QueryLanguage.Inputs.Query = this.CacheQueryPlugin.outputQuery;
        }
        #endregion LanguagePrediction


        #region SpellerProcessorPlugin
        [Timeout("*", 350)]
        public virtual global::Xap.ISpellerProcessorPlugin_10_ForInheritance SpellerProcessorPlugin { get; } = WorkflowServices.CreateInstance<global::Xap.ISpellerProcessorPlugin_10_ForInheritance>();
        protected virtual void BindSpellerProcessorPlugin()
        {
            this.SpellerProcessorPlugin.Inputs.query = this.CacheQueryPlugin.outputQuery;
            this.SpellerProcessorPlugin.Inputs.spellerAnswerResponse = this.SpellerCombiner.spellerAnswerCombinedKif.Cast<global::Speller.AlterationProviderResponse_1, global::QU.AnnotationProviderResponse_1>();
        }
        #endregion SpellerProcessorPlugin

        #region DolphinAdapterWorkflow
        public virtual global::DolphinAdapter.Workflows.IDolphinAdapterWorkflow_ForInheritance DolphinAdapterWorkflow { get; } = WorkflowServices.CreateInstance<DolphinAdapter.Workflows.IDolphinAdapterWorkflow_ForInheritance>();
        protected virtual void BindDolphinAdapterWorkflow()
        {
            this.DolphinAdapterWorkflow.Inputs.dolphinParseResult = this.QueryParserV3.resultV3;
            this.DolphinAdapterWorkflow.Inputs.orcaResult = this.ORCADeepIntentWorkflow.orcaResult;
            this.DolphinAdapterWorkflow.Inputs.dolphinLUDomains = this.QueryParserV3.LUDomains;
            this.DolphinAdapterWorkflow.Inputs.queryIndexResult = this.QueryIndex.preWebV3PrecisionOutput;
            this.DolphinAdapterWorkflow.Inputs.qasResponse = this.XapQuServiceAnswer.queryRepresentationResponse;
            this.DolphinAdapterWorkflow.Inputs.qasSpellerResponse = this.QASWithSpeller.queryRepresentationResponse;
        }
        #endregion


        #region LocationShift
        [Timeout("*", 40)]
        public virtual global::LocationShift.ILocationShiftWorkflow_1_ForInheritance LocationShift { get; } = WorkflowServices.CreateInstance<global::LocationShift.ILocationShiftWorkflow_1_ForInheritance>();
        protected virtual void BindLocationShift()
        {
            this.LocationShift.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.LocationShift.Inputs.instrumentationData = this.InstrumentationData;
            this.LocationShift.Inputs.rawQuery = this.CacheQueryPlugin.outputQuery;
            this.LocationShift.Inputs.userIdentification = this.UserIdentification;
        }
        #endregion LocationShift


        #region HybridLes
        public virtual global::Local.Workflows.IFastLocationExtraction_17_ForInheritance HybridLes { get; } = WorkflowServices.CreateInstance<global::Local.Workflows.IFastLocationExtraction_17_ForInheritance>();
        protected virtual void BindHybridLes()
        {
            this.HybridLes.Inputs.Augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.HybridLes.Inputs.InstrumentationData = this.InstrumentationData;
            this.HybridLes.Inputs.LocalMode = WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"FirstPageResults"; });
            this.HybridLes.Inputs.QasQueryRepresentation = this.XapQuWithOMA.queryRepresentationResponse;
            this.HybridLes.Inputs.Query = this.CacheQueryPlugin.outputQuery;
            this.HybridLes.Inputs.ShiftedLatLong = this.LocationShift.latLong;
            this.HybridLes.Inputs.UserIdentification = this.UserIdentification;
        }
        #endregion HybridLes


        #region SpellerFLE
        [Timeout("*", 350)]
        public virtual global::Local.Workflows.IFastLocationExtractionSelector_3_ForInheritance SpellerFLE { get; } = WorkflowServices.CreateInstance<global::Local.Workflows.IFastLocationExtractionSelector_3_ForInheritance>();
        protected virtual void BindSpellerFLE()
        {
            this.SpellerFLE.Inputs.Augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.SpellerFLE.Inputs.InstrumentationData = this.InstrumentationData;
            this.SpellerFLE.Inputs.ns_idLookupLocations = this.HybridLes.idLookupLocations;
            this.SpellerFLE.Inputs.ns_lesKifResponse = this.HybridLes.lesKifResponse;
            this.SpellerFLE.Inputs.ns_lesResponse = this.HybridLes.lesResponse;
            this.SpellerFLE.Inputs.ns_locationContext = this.HybridLes.locationContext;
            this.SpellerFLE.Inputs.ns_qasResponse = this.HybridLes.qasResponse;
            this.SpellerFLE.Inputs.ns_rawImplicitLocations = this.HybridLes.rawImplicitLocations;
            this.SpellerFLE.Inputs.ns_unbasedExtractedLocations = this.HybridLes.unbasedExtractedLocations;
            this.SpellerFLE.Inputs.QasWithSpellerQueryRepresentation = this.QASWithSpeller.queryRepresentationResponse;
            this.SpellerFLE.Inputs.Query = this.CacheQueryPlugin.outputQuery;
            this.SpellerFLE.Inputs.ShiftedLatLong = this.LocationShift.latLong;
            this.SpellerFLE.Inputs.UserIdentification = this.UserIdentification;
            this.SpellerFLE.Inputs.ns_adsLogInstr = this.HybridLes.adsLogInstr;
        }
        #endregion SpellerFLE


        #region ORCADeepIntentWorkflow
        public virtual ORCA.Plugins.IORCAQueryLanguageConditionPlugin_1_ForInheritance ORCAQueryLanguageCondition { get; } = WorkflowServices.CreateInstance<ORCA.Plugins.IORCAQueryLanguageConditionPlugin_1_ForInheritance>();
        protected virtual void BindORCAQueryLanguageCondition()
        {
        }

        public virtual ORCA.Plugins.IORCAQueryLanguagePassthrough_1_ForInheritance ORCAQueryLanguagePassthrough { get; } = WorkflowServices.CreateInstance<ORCA.Plugins.IORCAQueryLanguagePassthrough_1_ForInheritance>();
        protected virtual void BindORCAQueryLanguagePassthrough()
        {
            ORCAQueryLanguagePassthrough.Inputs.queryLanguageInput = this.QueryLanguage.languagePredictionResponse;
        }

        public virtual ORCA.Workflows.IORCAWorkflow_2_ForInheritance ORCADeepIntentWorkflow { get; } = WorkflowServices.CreateInstance<ORCA.Workflows.IORCAWorkflow_2_ForInheritance>();
        protected virtual void BindORCADeepIntentWorkflow()
        {
            ORCADeepIntentWorkflow.Inputs.Query = this.CacheQueryPlugin.outputQuery;
            ORCADeepIntentWorkflow.Inputs.SpellerOutput = this.SpellerProcessorPlugin.spellerOutput;
            ORCADeepIntentWorkflow.Inputs.QueryLanguageResponse = this.ORCAQueryLanguagePassthrough.result;
            ORCADeepIntentWorkflow.Inputs.Augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            ORCADeepIntentWorkflow.Inputs.webResult = WebAnswer.webResults;
            ORCADeepIntentWorkflow.Inputs.QueryIndexPrecisionLayerOutputV2 = this.QueryIndex.preWebV2PrecisionOutput;
            ORCADeepIntentWorkflow.Inputs.ProcessedWebResponse = this.QueryIndex.processedWebResponse;
            ORCADeepIntentWorkflow.Inputs.DolphinAdapterResult = this.DolphinAdapterWorkflow.sparqlResult;
            ORCADeepIntentWorkflow.Inputs.BingQueryMTResult = this.BingQueryMTWorkflow.result;
            ORCADeepIntentWorkflow.Inputs.LesResponse = this.SpellerFLE.lesResponse;
            ORCADeepIntentWorkflow.Inputs.LesSpellerResponse = this.SpellerFLE.lesResponseSpeller;
        }
        #endregion







        #region WebAnswer
        [Timeout("*", 5000)]
        public virtual global::Platform.IWebAnswerComposer_3_ForInheritance WebAnswer { get; } = WorkflowServices.CreateInstance<global::Platform.IWebAnswerComposer_3_ForInheritance>();
        protected virtual void BindWebAnswer()
        {
            this.WebAnswer.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.WebAnswer.Inputs.query = this.CacheQueryPlugin.outputQuery;
            // this.WebAnswer.Inputs.query = this.Query;
            // this.WebAnswer.Inputs.augmentsAndVariants = WorkflowServices.CreateDataSource<global::Platform.LegacyPluginAugmentsAndVariantsData>(t => { t.Augmentation = @"[AdService EnableORV=""1""][BAH Workflow=""Workflow_APlusNext""]"; });
            this.WebAnswer.Inputs.augmentsAndVariants = this.CrossLangAugmentPlugin.modifiedUserAugmentations;

            this.WebAnswer.Inputs.retryEnabled = this.isNoResultsEnabled.output;
            this.WebAnswer.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });

            this.WebAnswer.Inputs.dependents = new Aggregate<global::Platform.StringData> { WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"QueryWordBagGen"; }) };
            this.WebAnswer.Inputs.dependents.Add(WorkflowServices.CreateDataSource<global::Platform.StringData>(t => { t.Value = @"BingMeAnnotation"; }));



            this.WebAnswer.Inputs.instrumentationData = this.InstrumentationData;
            this.WebAnswer.Inputs.serviceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"WebAnswer"; });
            this.WebAnswer.Inputs.userIdentification = this.UserIdentification;
            this.WebAnswer.Inputs.virtualServiceName = WorkflowServices.CreateDataSource<global::Platform.ServiceNameData>(t => { t.ServiceName = @"WebAnswer"; });


            this.WebAnswer.Inputs.depInputAqrs = new Aggregate<global::Platform.LegacyQueryResponseData> { this.CombinedAlterationService.output, this.QRAggregator.alterationOutputAQR, this.SpellerAnswer.output, this.XapQuAthena.output, this.XapQuServiceAnswer.output, this.XapQuWithOMA.output };
            this.WebAnswer.Inputs.depInputAqrs.Add(this.CqnaPreweb.outputForWebAnswer);
            this.WebAnswer.Inputs.depInputAqrs.Add(this.QASMigrationWorkflow.aqr_http_output);
            this.WebAnswer.Inputs.depInputAqrs.Add(this.ContextualQLFMain.ContextualQLFOutput);
            this.WebAnswer.Inputs.depInputAqrs.Add(this.FlightAssignmentServiceWithSegmentRelevanceLBFA.Output);
            this.WebAnswer.Inputs.depInputAqrs.Add(this.ORCADeepIntentWorkflow.adoContainer);

        }
        #endregion WebAnswer


        #region AQRCombiner
        [Timeout("*", 10000)]
        public virtual global::Xap.IFinalAQRCombinerPlugin_1_ForInheritance AQRCombiner { get; } = WorkflowServices.CreateInstance<global::Xap.IFinalAQRCombinerPlugin_1_ForInheritance>();
        protected virtual void BindAQRCombiner()
        {
            this.AQRCombiner.Inputs.augmentations = this.CrossLangAugmentPlugin.modifiedAugmentations;
            this.AQRCombiner.Inputs.depInputAqrs = new Aggregate<global::Platform.LegacyQueryResponseData> { this.WebAnswer.webResponseAqr };
            this.AQRCombiner.Inputs.enabled = WorkflowServices.CreateDataSource<global::Platform.ExpressionResult>(t => { t.Expression = true; });
        }
        #endregion AQRCombiner


        #region AnswerLoggerPlugin
        public virtual CrossLangCache.Plugins.IAnswerLoggerPlugin_ForInheritance AnswerLoggerPlugin { get; } = WorkflowServices.CreateInstance<CrossLangCache.Plugins.IAnswerLoggerPlugin_ForInheritance>();
        protected virtual void BindAnswerLoggerPlugin()
        {
            this.AnswerLoggerPlugin.Inputs.inputQuery = this.CacheQueryPlugin.outputQuery;
            this.AnswerLoggerPlugin.Inputs.input = this.WebAnswer.webResponseAqr;
        }
        #endregion




    }
}
