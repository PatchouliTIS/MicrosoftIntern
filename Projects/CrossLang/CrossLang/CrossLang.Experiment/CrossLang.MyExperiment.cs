namespace Xap.Shared.BFPR
{
    using System;
    using System.Collections.Generic;
    using Xap.ExperimentFramework;
    using System.IO;
    using System.Security.Cryptography;

    public class BFPR : IExperiment
    {
        public Version Version
        {
            get
            {
                var thisAssemblyPath = Path.GetDirectoryName(typeof(BFPR).Assembly.Location);
                int major = int.Parse(File.ReadAllText(Path.Combine(thisAssemblyPath, "MajorVersion.txt")));
                int minor = int.Parse(File.ReadAllText(Path.Combine(thisAssemblyPath, "MinorVersion.txt")));
                return new Version(major, minor);
            }
        }



        public IList<IExperimentWorkflow> Workflows { get; }
        public IList<IExperimentDataStore> DataStores { get; } = new List<IExperimentDataStore>();

        public BFPR()
        {
            this.Workflows = new List<IExperimentWorkflow>()
            {
                new ExperimentWorkflow<Xap.Shared.CrossLang.ICrossLangWorkflow_1>(),
                new ExperimentWorkflow<Xap.Shared.CrossLang.ICrossLangCache_1>(),
                new ExperimentWorkflow<Xap.IBingFirstPageResults_69>(),
                new ExperimentWorkflow<Assistant.IBingFirstPageResults_34>(),
                new ExperimentWorkflow<WindowsPhone8.IBingFirstPageResults_40>(),
                new ExperimentWorkflow<Maps.IBingFirstPageResults_6>(),
                new ExperimentWorkflow<WinSearchCharm.IBingFirstPageResults_139>(),
                new ExperimentWorkflow<LocalAnswerExtensions.IHotelsLERProxyWorkflow>(),
                new ExperimentWorkflow<TravelTimeAnswer.IELAFlowWorkflow_1>(),
                new ExperimentWorkflow<TravelTimeAnswer.IPlanTripDynamicWorkflow_1>(),
                new ExperimentWorkflow<Ads.PaidSearchForNative.IPaidSearchForNativeMainV2_56>(),
                new ExperimentWorkflow<Recipes.EntitySegments.IWorkflow_DynamicSchemeProcessor_1>(),
                new ExperimentWorkflow<SportsLite.IUTWorkflow_9>(),
                new ExperimentWorkflow<SportsContainer.V2.IUTWorkflow_1>(),
                new ExperimentWorkflow<WPOAdsRealEstate.IPriceTrendWorkflow_1>(),
                new ExperimentWorkflow<MMChannel.Workflows.IMainWorkflow_3>(),
                new ExperimentWorkflow<MMPlaylist.IMainWorkflow_2>(),
                new ExperimentWorkflow<MMMixRanking.Workflows.IMainWorkflow_1>(),
                new ExperimentWorkflow<Multimedia.StructuredIndex.IWebCDGPointRequestWorkflowBond_4>(),
                new ExperimentWorkflow<DirectionAnswer.IDirectionAnswerPostWebWorkflow_3>(),
                new ExperimentWorkflow<AAPAnswer.IAttractionsCarouselRecommendWorkflow_1>(),
                new ExperimentWorkflow<RichDiscussions.Workflows.IRichDiscussionWorkflow_5>(),
                new ExperimentWorkflow<Recipes.IRecipeAnswerWorkflow_1>(),
                new ExperimentWorkflow<Recipes.ICarouselAnswerWorkflow_1>(),
                new ExperimentWorkflow<Multimedia.SharedPlugins.Workflows.IGenerateAnswerId>(),
                new ExperimentWorkflow<EntityIdLookUpAnswer.IRealEstatePostWebWrapper_1>(),
                new ExperimentWorkflow<XapPartnerPlugin.Events.DDAG.IBackendQueryWorkflow_5>(),
                new ExperimentWorkflow<LocalEntityRetrieval.ILiteIntegrationWorkflow>(),
                new ExperimentWorkflow<Local.IGeocodingRequery_1>(),
            };
        }
    }
}
