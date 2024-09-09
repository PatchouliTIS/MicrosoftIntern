namespace Xap.Shared.CrossLang
{
#pragma warning disable 4014
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using Xap.ComponentFramework;
    using Xap.WorkflowFramework;
    using Platform;
    using Local;

    [MajorVersionForLipstick(1)]
    [EnableLegacyCache(true)]
    [Timeout(@"*", 180000)]
    public class CrossLangWorkflow_1 : Xap.BingFirstPageResults_69
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2214: DoNotCallOverridableMethodsInConstructors", Justification = "CA2214: Warns about the possibility of derived classes not being initialized and that's not the case for XAP's Workflow C#.")]
        public CrossLangWorkflow_1() : base()
        {
            base.CTEXSuggestionBuilder.SetImplementation(null);
            base.RMValidation.SetImplementation(null);
            base.AdExternalProvsService.SetImplementation(null);
            base.RememberSerpQuery.SetImplementation(null);
            base.ElaCqna.SetImplementation(null);
            base.baseAds.SetImplementation(null);
            base.ConRegionAdBlocks.SetImplementation(null);
            base.PDSCRHints.SetImplementation(null);
            base.PDSFeatureAggregation.SetImplementation(null);
            base.PDSRulesWorkflow.SetImplementation(null);
            base.ZHCNAdsCondition.SetImplementation(null);
            base.adsCombination.SetImplementation(null);
            base.adsAqrZHCN.SetImplementation(null);
            base.adsAqr.SetImplementation(null);
            base.AdsDeltaAQR.SetImplementation(null);
            base.AdsPublisherSignalsProducer.SetImplementation(null);
            base.AdsWPOController.SetImplementation(null);
            base.BilingualDictionarySerpTranslate.SetImplementation(null);
            base.NewsAnswerV2WithQueryStats.SetImplementation(null);
            base.NewsClassifier.SetImplementation(null);
            base.NewsTrendingTopics.SetImplementation(null);
            base.CrispPlugin_DeDupeVideoCaptions.SetImplementation(null);
            base.CrispWorkflow_SocialChildAnswer.SetImplementation(null);
            base.CrispPlugin_Workflow_LocalChildAnswer.SetImplementation(null);
            base.CrispPlugin_Workflow_CrispImage.SetImplementation(null);
            base.CrispPlugin_GetAdultFlag.SetImplementation(null);
            base.CrispPlugin_Workflow_ImagesList.SetImplementation(null);
            base.GenImageCaptionCompositionSettings.SetImplementation(null);
            base.MultimediaKifImageAnswerWithQASFilter.SetImplementation(null);
            base.MultimediaKifVideoAnswer.SetImplementation(null);
            base.locationRanker.SetImplementation(null);
            base.ImplicitLocalXAP.SetImplementation(null);
            base.movieShowtimesTrigger.SetImplementation(null);
            base.movieShowtimesDebug.SetImplementation(null);
            base.movieShowtimesAnswerV3.SetImplementation(null);
            base.movieTitleShowtimesAnswer.SetImplementation(null);
            base.BepAnswerFetchInputs.SetImplementation(null);
            base.BepQasResponseProcessor.SetImplementation(null);
            base.MusicSerpWorkflow.SetImplementation(null);
            base.OfficeWebAppAnswer.SetImplementation(null);
            base.OnlineGamesWorkflow.SetImplementation(null);
            base.phonebookQuConsolidator.SetImplementation(null);
            base.PopularNow.SetImplementation(null);
            base.WeatherAnswerWithLocationExtractorAndQAS.SetImplementation(null);
            base.WeatherAnswerWithPlacesEP.SetImplementation(null);
            base.VINAnswerWithQAS.SetImplementation(null);
            base.TVProgramAnswer.SetImplementation(null);
            base.TVListingsAnswerDotNetWithQAS.SetImplementation(null);
            base.TravelCollageAnswerWithPlacesEP.SetImplementation(null);
            base.TravelAnswerWithQASLES.SetImplementation(null);
            base.TravelAnswerWithPlacesEP.SetImplementation(null);
            base.TranslateThisAnswer.SetImplementation(null);
            base.TranslateThis_TTAWorkflow.SetImplementation(null);
            base.TimeZoneAnswerWithQAS.SetImplementation(null);
            base.TianQiAnswer.SetImplementation(null);
            base.SportsAnswerWithQAS.SetImplementation(null);
            base.SongsAnswerDotNetWithQAS.SetImplementation(null);
            base.SoccerAnswerWithQAS.SetImplementation(null);
            base.SnowReportAnswerWithQAS.SetImplementation(null);
            base.SlideShowAnswerWithQAS.SetImplementation(null);
            base.SlideShowAnswerWithPlacesEP.SetImplementation(null);
            base.ShowtimesAnswerWithQAS.SetImplementation(null);
            base.ShoppingRelatedSearchesSerpAnswer.SetImplementation(null);
            base.RecipeWithQAS.SetImplementation(null);
            base.RealityTVAnswerWithQAS.SetImplementation(null);
            base.PrayerTimesAnswer.SetImplementation(null);
            base.PlacesAnswer.SetImplementation(null);
            base.PhonebookAnswerV2.SetImplementation(null);
            base.PackageTrackingAnswerWithQAS.SetImplementation(null);
            base.OfTheDayAnswer.SetImplementation(null);
            base.NoCodeAnswerWithPlacesEP.SetImplementation(null);
            base.NoCodeAnswer.SetImplementation(null);
            base.LotteryGBAnswer.SetImplementation(null);
            base.KhanVideoAnswerWithQAS.SetImplementation(null);
            base.GamesAnswerWithQAS.SetImplementation(null);
            base.CurrencyAnswerWithQAS.SetImplementation(null);
            base.CricketAnswer.SetImplementation(null);
            base.CommerceQueryTranslator.SetImplementation(null);
            base.CommerceAnswerV2.SetImplementation(null);
            base.BlackBoxAnswer.SetImplementation(null);
            base.AutosAnswerV2.SetImplementation(null);
            base.AttractionsAnswerWithQAS.SetImplementation(null);
            base.AttractionsAnswerV2WithQAS.SetImplementation(null);
            base.AttractionsAnswerV2WithPlacesEP.SetImplementation(null);
            base.AlphaUrlAnswerWithQAS.SetImplementation(null);
            base.AlphaAnswerWithQAS.SetImplementation(null);
            base.AlbumsAnswerDotNetWithQAS.SetImplementation(null);
            base.AFLAnswer.SetImplementation(null);
            base.YouPlusOutputIdentifier.SetImplementation(null);
            base.YouPlusCacheWriteAndFql.SetImplementation(null);
            base.YouPlus.SetImplementation(null);
            base.XapNetFacebookPeople.SetImplementation(null);
            base.Workflow_OpalSearch.SetImplementation(null);
            base.Workflow_AppChildAnswer.SetImplementation(null);
            base.WeatherRelatedAnswers.SetImplementation(null);
            base.WeatherMapsDebugInfoToADOConverter.SetImplementation(null);
            base.WeatherMapsDebugInfoCombiner.SetImplementation(null);
            base.WeatherMapsDebugAnswerCreateAQR.SetImplementation(null);
            base.WeatherMapsDarwinRankedContentCreator.SetImplementation(null);
            base.WeatherMaps.SetImplementation(null);
            base.WeatherLocationProcessor.SetImplementation(null);
            base.WeatherDebugQueryEvaluator.SetImplementation(null);
            base.trainsBetweenStations.SetImplementation(null);
            base.TopicalAuthoritiesAnswer.SetImplementation(null);
            base.TipCalculatorWrapper.SetImplementation(null);
            base.TipAnswer.SetImplementation(null);
            base.TimeZoneAnswer.SetImplementation(null);
            base.TimelineAnswerForSerp.SetImplementation(null);
            base.TechHelpAnswer.SetImplementation(null);
            base.TechGenericDIAAnswerWorkflow.SetImplementation(null);
            base.TechGenericDecider.SetImplementation(null);
            base.TechGenericAnswerWorkflow.SetImplementation(null);
            base.TechGenericAnswerPostWebWorkflow.SetImplementation(null);
            base.TechdownloadTaskPane.SetImplementation(null);
            base.SportsTennisAnswerWithQAS.SetImplementation(null);
            base.SportsTeamScheduleAnswer.SetImplementation(null);
            base.SportsSoccerAnswerWithQAS.SetImplementation(null);
            base.SportsScheduleAQR.SetImplementation(null);
            base.SportsNHLAnswerWithAlterations.SetImplementation(null);
            base.SportsNFLAnswerWithQAS.SetImplementation(null);
            base.SportsNBAAnswerWithQAS.SetImplementation(null);
            base.SportsMLBAnswerWithQAS.SetImplementation(null);
            base.SportsGolfAnswerWithQAS.SetImplementation(null);
            base.SportsEntityAnswerWithQAS.SetImplementation(null);
            base.SportsCricketAnswerWithQAS.SetImplementation(null);
            base.SportsCFBAnswerWithQAS.SetImplementation(null);
            base.SportsCBKAnswerWithQAS.SetImplementation(null);
            base.SportsAutoRacesAnswerWithQAS.SetImplementation(null);
            base.SportsAnswerLite.SetImplementation(null);
            base.SportsAnswerContainer.SetImplementation(null);
            base.SportsAFLAnswerWithQAS.SetImplementation(null);
            base.SocialVoiceCall.SetImplementation(null);
            base.SocialPostAnswer.SetImplementation(null);
            base.SocialImageAnswer_vNext.SetImplementation(null);
            base.SocialImageAnswer.SetImplementation(null);
            base.RewardsIntlAnswer.SetImplementation(null);
            base.RewardsAnswer.SetImplementation(null);
            base.MapsAnswerChildAnnotations.SetImplementation(null);
            base.locLogging.SetImplementation(null);
            base.LesMapsView.SetImplementation(null);
            base.HealthConditionsAnswer.SetImplementation(null);
            base.HealthAnswerV2.SetImplementation(null);
            base.FoodAnswerV1.SetImplementation(null);
            base.FltStatRelAnsAnnotationAnswer.SetImplementation(null);
            base.FitnessAnswer.SetImplementation(null);
            base.CalculatorZHCNPostProcessor.SetImplementation(null);
            base.BlackFridayStoresAnswer.SetImplementation(null);
            base.BlackFridaySingleStoreAnswer.SetImplementation(null);
            base.BlackFridayDealsAnswer.SetImplementation(null);
            base.applinkingWorkflow.SetImplementation(null);
            base.AppNSearch.SetImplementation(null);
            base.ChitChatAnswerV2_ChitChatAnswerWorkflow.SetImplementation(null);
            base.HybridLes.SetImplementation(null);
            base.InstantAnswers.SetImplementation(null);
            base.ElectionMainWF.SetImplementation(null);
            base.ActionsSideBySide.SetImplementation(null);
            base.CreateFirebirdAqr.SetImplementation(null);
            base.Deeplinks_Workflow_DeeplinksPreWeb.SetImplementation(null);
            base.queryIntentVideos.SetImplementation(null);
            base.slackLESInfo.SetImplementation(null);
            base.WeatherMapsFeature.SetImplementation(null);
            base.ImplicitLocalCounterfactualWorkflow.SetImplementation(null);
        }



        public new void ExecuteForInheritance()
        {
            this.Execute(
                output: out this.output,
                searchCacheStatus: out this.searchCacheStatus,
                aqrResponse: out this.aqrResponse,
                rankedContent: out this.rankedContent,
                ANNTrigerStatus: out this.ANNTrigerStatus,
                Query: this.Query,
                UserIdentification: this.UserIdentification,
                InstrumentationData: this.InstrumentationData,
                Augmentations: this.Augmentations,
                LegacyCacheEnabled: this.LegacyCacheEnabled,
                aqmDeviceCapabilities: this.aqmDeviceCapabilities,
                pbaKifRequest: this.pbaKifRequest,
                pbaOptions: this.pbaOptions,
                LocationContext: this.LocationContext,
                EntityApiRequestArguments: this.EntityApiRequestArguments,
                ParameterizedRequests: this.ParameterizedRequests,
                cuRequestData: this.cuRequestData,
                lgRequestData: this.lgRequestData,
                Entities: this.Entities,
                adsRequest: this.adsRequest,
                RewardsParameters: this.RewardsParameters,
                answerTypeFilters: this.answerTypeFilters,
                request: this.request);
        }
        public new virtual void Execute(
            out Task<global::Platform.LegacyQueryResponseData> output,
            out Task<global::Platform.BoolData> searchCacheStatus,
            out Task<global::Platform.LegacyQueryResponseData> aqrResponse,
            out Task<global::AnswersRanker.RankedContent_1> rankedContent,
            out Task<global::SpaceV.ANNTrigerStatus> ANNTrigerStatus,
            Task<global::Platform.Query> Query = null,
            Task<global::Platform.UserIdentification> UserIdentification = null,
            Task<global::Platform.InstrumentationData> InstrumentationData = null,
            Task<global::Platform.Augmentations> Augmentations = null,
            Task<global::Platform.StringData> LegacyCacheEnabled = null,
            Task<global::DeviceCapabilities.DeviceCapabilities_1> aqmDeviceCapabilities = null,
            Task<global::Local.PhonebookRequest_1> pbaKifRequest = null,
            Task<global::Local.Pba.Options_3> pbaOptions = null,
            Task<global::Local.LocationContext_1> LocationContext = null,
            Task<global::Entities.Request.Custom.RequestArgs> EntityApiRequestArguments = null,
            Task<global::Answer.Shared.ParameterizedRequestCollection_1> ParameterizedRequests = null,
            Task<global::Cortana.CU.RequestData> cuRequestData = null,
            Task<global::Cortana.LG.RequestData> lgRequestData = null,
            Task<global::Entities.Request.EntityRequestSettings_1> Entities = null,
            Task<global::Ads.RequestAugments.Request> adsRequest = null,
            Task<global::RewardsModern.RewardsBag_1> RewardsParameters = null,
            Task<global::Answer.Shared.AnswerTypeFilters_1> answerTypeFilters = null,
            Task<global::Multimedia.FavReq.RequestWrapper_4> request = null)
        {
            this.Query = Query;
            this.UserIdentification = UserIdentification;
            this.InstrumentationData = InstrumentationData;
            this.Augmentations = Augmentations;
            this.LegacyCacheEnabled = LegacyCacheEnabled;
            this.aqmDeviceCapabilities = aqmDeviceCapabilities;
            this.pbaKifRequest = pbaKifRequest;
            this.pbaOptions = pbaOptions;
            this.LocationContext = LocationContext;
            this.EntityApiRequestArguments = EntityApiRequestArguments;
            this.ParameterizedRequests = ParameterizedRequests;
            this.cuRequestData = cuRequestData;
            this.lgRequestData = lgRequestData;
            this.Entities = Entities;
            this.adsRequest = adsRequest;
            this.RewardsParameters = RewardsParameters;
            this.answerTypeFilters = answerTypeFilters;
            this.request = request;




            this.BindVariantSetPlugin();
            this.VariantSetPlugin.ExecuteForInheritance();


            // TODO: QueryModify Before Base Init
            this.BindQueryModifyPlugin();
            this.QueryModifyPlugin.ExecuteForInheritance();
            // this.Query = this.QueryModifyPlugin.modifiedQuery;


            // Bind Output to base
            this.BindCrossLangQueryBaseInput();
            base.ExecuteForInheritance();

            this.BindDisplayResultBondPlugin();
            this.DisplayResultBondPlugin.ExecuteForInheritance();


            this.BindAggregateARQ();
            this.CreateAggregateARQ.ExecuteForInheritance();


            this.BindAggregateFinalResponsePlugin();
            this.AggregateFinalResponsePlugin.ExecuteForInheritance();


            // output = base.output;
            output = this.AggregateFinalResponsePlugin.output;
            searchCacheStatus = base.searchCacheStatus;
            aqrResponse = base.DateCalcCreateAqr.output;
            rankedContent = base.AgeDateCalculator_DarwinFastRankOutputter.rankedContent;
            ANNTrigerStatus = base.ANNResponseProcessor.ANNTrigerStatus;
        }

        protected override void BindXapPartnerPlugin_Events_EventsWorkflow()
        {
            base.BindXapPartnerPlugin_Events_EventsWorkflow();
            this.XapPartnerPlugin_Events_EventsWorkflow.Inputs.webResults = WebAnswer.webResults;
        }

        #region WebAnswer
        protected override void BindWebAnswer()
        {
            base.BindWebAnswer();
            this.WebAnswer.Inputs.augmentsAndVariants = this.VariantSetPlugin.modifiedUserAugmentations;
        }
        #endregion WebAnswer


        #region VariantSetPlugin
        [Timeout("*", 10000)]
        public virtual IVariantSetPlugin_ForInheritance VariantSetPlugin { get; } = WorkflowServices.CreateInstance<IVariantSetPlugin_ForInheritance>();

        protected virtual void BindVariantSetPlugin()
        {
            this.VariantSetPlugin.Inputs.query = this.Query;
            this.VariantSetPlugin.Inputs.augmentations = this.Augmentations;
        }
        #endregion


        #region QueryModifyPlugin
        [Timeout("*", 120000)]
        public virtual IQueryModifyPlugin_ForInheritance QueryModifyPlugin { get; } = WorkflowServices.CreateInstance<IQueryModifyPlugin_ForInheritance>();

        protected virtual void BindQueryModifyPlugin()
        {
            this.QueryModifyPlugin.Inputs.query = this.VariantSetPlugin.modifiedQuery;
            this.QueryModifyPlugin.Inputs.augmentations = this.VariantSetPlugin.modifiedAugmentations;
        }
        #endregion

        #region BindCrossLangQueryBaseInput
        protected virtual void BindCrossLangQueryBaseInput()
        {
            this.Query = this.QueryModifyPlugin.modifiedQuery;
            this.Augmentations = this.VariantSetPlugin.modifiedAugmentations;
        }
        #endregion


        #region DisplayResultBondPlugin
        [Timeout("*", 10000)]
        public virtual IDisplayResultBondPlugin_ForInheritance DisplayResultBondPlugin { get; } = WorkflowServices.CreateInstance<IDisplayResultBondPlugin_ForInheritance>();

        protected virtual void BindDisplayResultBondPlugin()
        {
            this.DisplayResultBondPlugin.Inputs.query = this.QueryModifyPlugin.modifiedQuery;
        }
        #endregion DisplayResultBondPlugin


        #region AggregateARQ
        public virtual global::Platform.ICreateBondAQR_2_ForInheritance<global::CrossLang.QueryDisplay> CreateAggregateARQ { get; } = WorkflowServices.CreateInstance<global::Platform.ICreateBondAQR_2_ForInheritance<global::CrossLang.QueryDisplay>>();

        protected virtual void BindAggregateARQ()
        {
            this.CreateAggregateARQ.Inputs.answers = new Aggregate<global::CrossLang.QueryDisplay> { this.DisplayResultBondPlugin.output };
            this.CreateAggregateARQ.Inputs.scenarioName = this.DisplayResultBondPlugin.scenarioName;
            this.CreateAggregateARQ.Inputs.serviceName = this.DisplayResultBondPlugin.serviceName;
        }
        #endregion AggregateARQ



        #region AggregateFinalResponsePlugin
        public virtual IAggregateFinalResponsePlugin_ForInheritance AggregateFinalResponsePlugin { get; } = WorkflowServices.CreateInstance<IAggregateFinalResponsePlugin_ForInheritance>();
        protected virtual void BindAggregateFinalResponsePlugin()
        {
            this.AggregateFinalResponsePlugin.Inputs.baseData = base.output;
            this.AggregateFinalResponsePlugin.Inputs.addData = CreateAggregateARQ.output;
        }
        #endregion BindFinalResponseOutput

    }
}
