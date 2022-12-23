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
            if (args.Count() < 1)
            {
                Console.WriteLine("ListModules.exe <user alias>");
            }
            else
            {
                GetModuleList(args[0]);
            }
        }

        static void GetModuleList(string alias)
        {
            var environment = new AetherEnvironment(endpointAddress: AetherEnvironment.Aether1PEndpointAddress, clientName: "ListAetherModules");

            IAetherSearchEntityResponse<IAetherSearchExecutable> searchResult = environment.ExecutableSearch("owner:" + alias + " count:500");

            StreamWriter sw = new StreamWriter(alias + "_modules.txt");

            int module_cnt = 0;
            while (searchResult.Entities.Count() > 0)
            {
                foreach (var entity in searchResult.Entities)
                {
                    List<string> required_attributes = new List<string> { 
                        entity.Name, 
                        entity.Version, 
                        entity.Id, 
                        entity.FamilyId, 
                        entity.ResourceTypeName, 
                        entity.CreatedDate.ToString(), 
                        entity.EntityStatus.ToString() 
                    };
                    string line = String.Join(" * ", required_attributes);
                    sw.WriteLine(line);
                }

                module_cnt += searchResult.Entities.Count();
                searchResult = environment.ExecutableSearch("owner:" + alias + " count:500 skip:" + module_cnt);
            }
            sw.Close();
        }
    }
}
