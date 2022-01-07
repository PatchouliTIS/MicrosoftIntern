using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Aether.Library;

namespace ListModules
{
    class Program
    {
        static void Main(string[] args)
        {
            get_modules(args[0]);
        }

        static void get_modules(string alias)
        {
            var environment = new AetherEnvironment(endpointAddress: AetherEnvironment.Aether1PEndpointAddress, clientName: "ListAetherModules");

            IAetherSearchEntityResponse<IAetherSearchExecutable> searchResult = environment.ExecutableSearch("owner:" + alias);

            StreamWriter sw = new StreamWriter(alias + "_modules.txt");
            foreach (var entity in searchResult.Entities)
            {
                List<string> required_attributes = new List<string> { entity.Name, entity.Version, entity.Id, entity.FamilyId, entity.ResourceTypeName, entity.CreatedDate.ToString(), entity.EntityStatus.ToString() };
                string line = String.Join(" * ", required_attributes);
                sw.WriteLine(line);
            }
            sw.Close();
        }
    }
}
