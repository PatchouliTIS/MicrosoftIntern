using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Aether.Library;

namespace RetrieveScheduledJobs
{
    class Program
    {
        static void Main(string[] args)
        {
            var environment = new AetherEnvironment(endpointAddress: AetherEnvironment.Aether1PEndpointAddress, clientName: "RetrieveScheduledJobs");

            String date_time = DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss");
            String query = args.Count() > 0 && args[0].Equals("galaxy") ? "/galaxyprod/ /jobDetails/" : "[scheduled by";
            String job = args.Count() > 0 && args[0].Equals("galaxy") ? "Galaxy" : "Aether";
            String filename = job + "_scheduled_jobs_" + date_time + ".txt";

            const int max_retrieve_cnt = 100000;
            GetScheduledJobs(environment, query, filename, max_retrieve_cnt);
        }

        static void GetScheduledJobs(AetherEnvironment environment, String query, String filename, int max_exp_cnt)
        {
            Console.WriteLine("Retieving scheduled jobs");
            Console.WriteLine(query);

            StreamWriter sw = new StreamWriter(filename);
            sw.WriteLine(String.Join("\t", new string[] {"ID", "Owner", "Team ID", "Experiment Type", "Experiment Status", "StartTime", "EndTime", "Description" }));

            // retrieve experiments with schduled by keyword
            IAetherSearchEntityResponse<IAetherSearchExperiment> searchResult = environment.ExperimentSearch(query + " count:500");

            Console.WriteLine();

            int expcnt = 0;
            while (searchResult.Entities.Count() > 0)
            {
                Console.Write("\rGetting " + expcnt);
                foreach (var entity in searchResult.Entities)
                {
                    List<string> required_attributes = new List<string> {
                        entity.Id,
                        entity.Owner,
                        entity.TeamId,
                        entity.ExperimentType.ToString(),
                        entity.CachedStatus.ToString(),  // call GetStatus to get latest status
                        entity.DateStart.ToString(),
                        entity.DateEnd.ToString(),
                        entity.Description
                    };
                    string line = String.Join("\t", required_attributes);
                    sw.WriteLine(line);
                }

                expcnt += searchResult.Entities.Count();

                if (expcnt >= max_exp_cnt)
                {
                    break;
                }

                searchResult = environment.ExperimentSearch(query + " count:500 skip:" + expcnt);
            }
            sw.Close();

            Console.WriteLine();
        }
    }
}
