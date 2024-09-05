using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xap.ExecutionFramework;
using Xap.ExecutionFramework.Extensions;
using Xap.PluginFramework;
using System.Runtime.InteropServices;
using Xap;

namespace Plugin.Tests
{
    [TestClass]
    public class PluginTests
    {

        // public const string tail = "Genshin Impact";

        [TestMethod]
        public void TestCacheQueryPlugin()
        {
            CrossLangCache.Plugins.ICacheQueryPlugin cacheQueryPlugin = ExecutionServices.CreateInstance<CrossLangCache.Plugins.ICacheQueryPlugin>();

            cacheQueryPlugin.SetEnvironment("DevMachine");
            cacheQueryPlugin.SetStaticVariant("MKT", "ZH-CN");

            Console.WriteLine("CacheQueryPlugin TESTING");
            var augmentations = ExecutionServices.CreateInstance<Platform.Augmentations>();
            var targetMKT = ExecutionServices.CreateInstance<Platform.StringData>();
            var pre = ExecutionServices.CreateInstance<global::QueryIndex.CanonicalQueriesOutputV3>();
            pre.RankedCanonicalQueries = new List<global::QueryIndex.CanonicalQueryCandidateV3>();
            var candidateV3 = ExecutionServices.CreateInstance<global::QueryIndex.CanonicalQueryCandidateV3>();
            candidateV3.CanonicalQuery = "how to park in summer palace";
            pre.RankedCanonicalQueries.Add(candidateV3);
            targetMKT.Value = "zh-cn";
            var intput_targetMKT = Task.FromResult(targetMKT);

            var input_pre = Task.FromResult(pre);

            var result = cacheQueryPlugin.Execute(input_pre, intput_targetMKT);

            Console.WriteLine("CacheQueryPlugin OUTPUT:{0}\n{1}", cacheQueryPlugin.Url.Result.Count(), cacheQueryPlugin.Url.Result.First().Value);

            Assert.IsTrue(result.Result.Success);
        }



    }
}
