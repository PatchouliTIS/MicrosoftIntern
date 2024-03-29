using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xap.ComponentFramework;
using Xap.PluginFramework;

namespace CrossLangCache.Plugins
{
    public class CacheQueryPlugin : IPlugin
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "0", Justification = "PluginServices guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "1", Justification = "Required inputs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "2", Justification = "Configs guaranteed non-null by ApplicationHost")]
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1062:Validate arguments of public methods", MessageId = "3", Justification = "Outputs guaranteed non-null by ApplicationHost")]
        public PluginResult Execute(PluginServices pluginServices,
                                    PluginOutput<global::Platform.Query> outputQuery,
                                    global::Platform.Query baseQuery,
                                    global::Platform.StringData targetMKT)
        {
            pluginServices.Logger.Info("ENTERING QueryCachePool");


            pluginServices.Logger.Info("Is BaseQuery Null:{0}", baseQuery == null);
            if (baseQuery == null)
            {
                return PluginResult.Succeeded;
            }

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


            ss.GetTransQuery(baseQuery.RawQuery, mkt, out string transQuery);
            if(transQuery != null)
            {
                pluginServices.Logger.Info("TRANS QUERY: " + transQuery);
                outputQuery.Data = pluginServices.CreateInstance<global::Platform.Query>();
                outputQuery.Data.RawQuery = transQuery;
                outputQuery.Data.NormalizedQuery = transQuery;
                outputQuery.Data.WordBrokenQuery = baseQuery.WordBrokenQuery;
                outputQuery.Data.WordBrokenToRawQueryMapping = baseQuery.WordBrokenToRawQueryMapping;
                outputQuery.Data.QueryLanguage = baseQuery.QueryLanguage;
            } else
            {
                outputQuery.Data = null;
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

        public void GetTransQuery(string query, string mkt, out string transQuery)
        {
            mkt = mkt.ToLower();
            query = query.ToLower();

            int similarQueryIndex = findMostSimilarString(query);
            if (similarQueryIndex == -1)
            {
                transQuery = null;
                return;
            }
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
