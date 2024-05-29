using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.PluginFramework;

namespace CrossLangCache.Plugins
{
    public class CacheQueryPlugin : IPlugin
    {

        public const string SERVICE_NAME = "crosslangsearch";
        public const string SCENARIO_NAME = "firstpage";


        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public PluginResult Execute(PluginServices pluginServices,
                                    PluginOutput<global::Platform.StringData> modifiedQuery,
                                    CollectionPluginOutput<IEnumerable<global::Platform.StringData>> Url,
                                    CollectionPluginOutput<IEnumerable<global::Platform.StringData>> Title,
                                    CollectionPluginOutput<IEnumerable<global::Platform.StringData>> Snippet,
                                    global::QueryIndex.CanonicalQueriesOutputV3 preWebV3PrecisionOutput = null,
                                    global::Platform.StringData targetMKT = null)
        {
            // Access Query
            string targetQuery = preWebV3PrecisionOutput.RankedCanonicalQueries.First().CanonicalQuery;
            pluginServices.Logger.Info("query access:{0}", targetQuery);

            // Similarity Match
            StringSimilarity ss = new StringSimilarity(pluginServices);

            string mkt;
            if (targetMKT == null)
            {
                pluginServices.Logger.Info("targetMKT is NULL");
                return PluginResult.Succeeded;
            }
            else
            {
                mkt = targetMKT.Value;
            }

            if (targetMKT.Value == null)
            {
                if (pluginServices.Variants.TryGetValue("MKT", out mkt))
                {
                    pluginServices.Logger.Info("Get MKT Succuss:" + mkt);

                }
                else
                {
                    pluginServices.Logger.Info("Get MKT Failed: No MKT exists");
                }
            }
            else
            {
                mkt = targetMKT.Value;
            }


            ss.GetTransQuery(targetQuery, mkt, out string transQuery, out int index);

            var modQuery = pluginServices.CreateInstance<Platform.StringData>();
            modifiedQuery.Data = modQuery;
            modifiedQuery.Data.Value = transQuery;
            pluginServices.Logger.Info("TransQuery :{0}", transQuery);

            Url.Data = new List<global::Platform.StringData>();
            var pair = ss.chnUrl[index];
            foreach (string s in pair)
            {
                var sd = pluginServices.CreateInstance<global::Platform.StringData>();
                sd.Value = s;
                Url.Data = Url.Data.Append(sd);
            }

            Snippet.Data = new List<global::Platform.StringData>();
            pair = ss.chnSnippet[index];
            foreach (string s in pair)
            {
                var sd = pluginServices.CreateInstance<global::Platform.StringData>();
                sd.Value = s;
                Snippet.Data = Snippet.Data.Append(sd);
            }

            Title.Data = new List<global::Platform.StringData>();
            pair = ss.chnTitle[index];
            foreach (string s in pair)
            {
                var sd = pluginServices.CreateInstance<global::Platform.StringData>();
                sd.Value = s;
                Title.Data = Title.Data.Append(sd);
            }

            return PluginResult.Succeeded;
        }


    }



    #region StringSimilarity
    /**
     * A simple string similarity algorithm using BiGram and inverted indexing 
     * to compare the user's query with the query in the Cache and select the 
     * query with the highest similarity and its corresponding translation.
     */
    public class StringSimilarity
    {
        private string[] queryPool;
        private string[] eng2chn;
        private string[] eng2jap;
        private string[] eng2fr;
        public Dictionary<int, List<string>> chnUrl;
        public Dictionary<int, List<string>> chnSnippet;
        public Dictionary<int, List<string>> chnTitle;
        private Dictionary<string, List<int>> invertedIndex;
        private PluginServices pluginServices;
        private const double THRESHOLD = 0.6;
        public StringSimilarity()
        {
            this.invertedIndex = new Dictionary<string, List<int>>();
            this.queryPool = new string[]
            {
                "how to park in summer palace",
                "sichuan famous cuisine",
                "interesting tourist attractions in nanjing",
                "how to park in fushimi inari shrine",
                "kyoto famous cuisine",
                "interesting tourist attractions in sapporo",
                "how to park in notre dame de paris",
                "eastern france famous cuisine",
                "interesting tourist attractions in paris"
            };

            this.eng2chn = new string[]
            {
                "如何在颐和园停车",
                "四川著名美食",
                "南京有趣的旅游景点",
                "如何在伏见稻荷大社停车",
                "京都著名美食",
                "札幌有趣的旅游景点",
                "如何在巴黎圣母院停车",
                "东法兰西著名美食",
                "巴黎有趣的旅游景点"
            };

            this.chnSnippet = new Dictionary<int, List<string>>
            {
                [0] = new List<string> { "步行到新建宫门300米，步行到东宫门450米。 新建宫门马路两侧划有停车线可停车。 首小时6元，第2个小时开始每小时9元。", "山水格局、主要景点都说到了，那么怎么设计游览路线呢： 路线一：全园打卡暴走（6-8小时/全天） 路线的起点是颐和园西门，临近地铁西郊线颐和园西门站，同时有一个面积较大的停车场，不论是公交出游还", "有停车场. 公交线路：374、74、437路. 5、颐和园西门. 有停车场. 地铁线路：西郊线. 公交线路：469、539路. 6、北如意门. 暂无公交、地铁、停车场. 温馨提示：微" },
                [1] = new List<string> { "推荐一：三大炮. 三大炮是著名的四川地区传统特色小吃，主要由糯米制成，由于在抛扔糯米团时，三大炮如“弹丸”一样，发出“当、当、当”的响声，分为“铁炮”，“火炮”，“枪炮”，故名总称“三大炮”。 来成都旅行，常规景点快窄巷子、锦里、洛带古镇都很容易见到三大炮。 坦率来说，本地人现在消 ", "四川美食取材广泛、调味多变，以麻辣著称。 在之前的文章里瘦猴就跟大家分享过一些四川的经典名菜，像麻婆豆腐、回锅肉、夫妻肺片、宫保鸡丁等等，都是四川", "四川的小吃包罗万象，小吃文化更是博大精深！ 下面小编给大家八一八四川那些让人欲罢不能特色小吃， 1.钵钵鸡. 钵钵鸡，流传至今有上百年的历史了，是以陶器钵盛放配以麻辣为主的配料，加上多种配料" }
            };

            this.chnTitle = new Dictionary<int, List<string>>
            {
                [0] = new List<string> { "颐和园自驾车停车场详解︱颐和园攻略七 - 百家号", "去颐和园有哪些实用攻略和好的建议？ - 知乎", "北京颐和园在哪里？怎么去？（公交+地铁+自驾+停车场" },
                [1] = new List<string> { "四川有哪些让人欲罢不能的小吃？ - 知乎", "四川最有名的12种名小吃，每种都是四川人的心头爱", "四川十大特色小吃，看着流口水 - 知乎" }
            };

            this.chnUrl = new Dictionary<int, List<string>>
            {
                [0] = new List<string> { @"https://baijiahao.baidu.com/s?id=1690502829713929114", @"https://baijiahao.baidu.com/s?id=1690502829713929114", @"http://bj.bendibao.com/tour/2018831/252059.shtm" },
                [1] = new List<string> { @"https://www.zhihu.com/question/24227485", @"https://zhuanlan.zhihu.com/p/33367027", @"https://baijiahao.baidu.com/s?id=1717317575157020876" }
            };

            this.eng2jap = new string[]
            {
                "夏宮で駐車する方法",
                "四川の有名な料理",
                "南京の面白い観光スポット",
                "伏見稲荷大社で駐車する方法",
                "京都の有名な料理",
                "札幌の面白い観光スポット",
                "ノートルダム大聖堂で駐車する方法",
                "東フランスの有名な料理",
                "パリの面白い観光スポット"
            };

            this.eng2fr = new string[]
            {
                "comment se garer au Palais d'été",
                "cuisine célèbre du Sichuan",
                "attractions touristiques intéressantes à Nanjing",
                "comment se garer au sanctuaire Fushimi Inari",
                "cuisine célèbre de Kyoto",
                "attractions touristiques intéressantes à Sapporo",
                "comment se garer à Notre-Dame de Paris",
                "cuisine célèbre de l'est de la France",
                "attractions touristiques intéressantes à Paris"
            };

            this.buildInvertedIndex();
        }

        public StringSimilarity(PluginServices ps) : this()
        {
            this.pluginServices = ps;
        }

        public void GetTransQuery(string query, string mkt, out string transQuery, out int index)
        {
            mkt = mkt.ToLower();
            query = query.ToLower();

            int similarQueryIndex = findMostSimilarString(query);
            if (similarQueryIndex == -1)
            {
                transQuery = null;
                index = -1;
                return;
            }

            index = similarQueryIndex;
            if (mkt.Equals("zh-cn"))
            {
                transQuery = this.eng2chn[similarQueryIndex];
            }
            else if (mkt.Equals("ja-jp"))
            {
                transQuery = this.eng2jap[similarQueryIndex];
            }
            else if (mkt.Equals("fr-fr"))
            {
                transQuery = this.eng2fr[similarQueryIndex];
            }
            else
            {
                transQuery = this.queryPool[similarQueryIndex];
            }

            return;
        }


        private int findMostSimilarString(string query)
        {
            int q = 2; // set the value of Q-gram according to task needs

            int mostSimilarStringIndex = 0;

            HashSet<string> queryQGramSet = generateQGramSet(query, q);
            Dictionary<int, int> similarityScores = new Dictionary<int, int>();

            foreach (string qGram in queryQGramSet)
            {
                if (invertedIndex.ContainsKey(qGram))
                {
                    foreach (int index in invertedIndex[qGram])
                    {
                        if (!similarityScores.ContainsKey(index))
                        {
                            similarityScores[index] = 0;
                        }
                        similarityScores[index]++;
                    }
                }
            }

            double maxScore = 0.0;
            // pick up the query string with the highest similarity
            foreach (KeyValuePair<int, int> pair in similarityScores)
            {
                // Calculate
                double currentScore = 2 * (double)pair.Value / (double)(queryPool[pair.Key].Length + query.Length);
                if (maxScore < currentScore)
                {
                    maxScore = currentScore;
                    mostSimilarStringIndex = pair.Key;
                }

                pluginServices.Logger.Info(this.queryPool[pair.Key] + "\t" + currentScore);

            }

            // use threshold to trigger cache workflow
            if (maxScore <= THRESHOLD)
            {
                mostSimilarStringIndex = -1;
            }

            return mostSimilarStringIndex;
        }


        private void buildInvertedIndex()
        {
            invertedIndex = new Dictionary<string, List<int>>();
            int index = 0;
            foreach (string poolString in this.queryPool)
            {
                HashSet<string> qGramSet = generateQGramSet(poolString, 2);
                foreach (string qGram in qGramSet)
                {
                    if (!invertedIndex.ContainsKey(qGram))
                    {
                        invertedIndex[qGram] = new List<int>();
                    }
                    invertedIndex[qGram].Add(index);
                }
                index++;
            }
        }

        private HashSet<string> generateQGramSet(string str, int q)
        {
            HashSet<string> qGramSet = new HashSet<string>();

            for (int i = 0; i <= str.Length - q; i++)
            {
                string qGram = str.Substring(i, q);
                qGramSet.Add(qGram);
            }

            return qGramSet;
        }



    }

    #endregion
}
