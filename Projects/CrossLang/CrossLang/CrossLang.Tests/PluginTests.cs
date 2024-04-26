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
        public void TestQueryModifyViaInterface()
        {
            // create query
            Xap.IQueryModifyPlugin queryModifyPlugin = ExecutionServices.CreateInstance<Xap.IQueryModifyPlugin>();
            queryModifyPlugin.SetEnvironment("DevMachine");
            queryModifyPlugin.SetStaticVariant("MKT", "ZH-CN");
            var query = ExecutionServices.CreateInstance<Platform.Query>(query => query.RawQuery = "How to park when visiting the Summer Palace");
            var input = Task.FromResult(query);
            var augmentations = ExecutionServices.CreateInstance<Platform.Augmentations>();
            var input_aug = Task.FromResult(augmentations);


            var result = queryModifyPlugin.Execute(input, input_aug);



            result.Wait();

            Console.WriteLine(">>> Test Done <<<");
            Console.WriteLine(result.Status);
            var pluginResult = result.Result;
            Console.WriteLine(queryModifyPlugin.modifiedQuery.Result.RawQuery);


            Assert.IsTrue(pluginResult.Success);
            Assert.IsFalse(queryModifyPlugin.GetExecutionContext().Exceptions.Any());
            
            // Assert.IsTrue(string.Equals(expectedQuery, queryModifyPlugin.modifiedQuery.Result.RawQuery));
        }

        [TestMethod]
        public void TestVariantSetPluginInterface()
        {
            Xap.IVariantSetPlugin variantSetPlugin = ExecutionServices.CreateInstance<Xap.IVariantSetPlugin>();
            variantSetPlugin.SetEnvironment("DevMachine");
            variantSetPlugin.SetStaticVariant("MKT", "ZH-CN");


            Console.WriteLine("VARIANT TESTING");
            var query = ExecutionServices.CreateInstance<Platform.Query>(query => query.RawQuery = "Tips for visiting the Summer Palace");
            var augmentations = ExecutionServices.CreateInstance<Platform.Augmentations>();
            var input_query = Task.FromResult(query);
            var input_aug = Task.FromResult(augmentations);

            var result = variantSetPlugin.Execute(input_query, input_aug);
           

            Assert.IsTrue(result.Result.Success);


        }
    }
}
