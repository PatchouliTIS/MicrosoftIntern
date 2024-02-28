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
            var query = ExecutionServices.CreateInstance<Platform.Query>(query => query.RawQuery = "Tips for visiting the Summer Palace");
            var augmentations = ExecutionServices.CreateInstance<Platform.Augmentations>();
            var targetMKT = ExecutionServices.CreateInstance<Platform.StringData>("zh-cn");
            var input_query = Task.FromResult(query);
            var input_augmentations = Task.FromResult(augmentations);
            var intput_targetMKT = Task.FromResult(targetMKT);

            var result = cacheQueryPlugin.Execute(input_query, input_augmentations, intput_targetMKT);

            Console.WriteLine(cacheQueryPlugin.outputQuery.Result.RawQuery);

            Assert.IsTrue(result.Result.Success);
        }
    }
}
