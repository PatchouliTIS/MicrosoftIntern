using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ModuleUsageStat;

namespace ConsolidateStat
{
    internal class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("ConsolidateStat.exe <user list> <stat file>");
                return;
            }

            string[] user_list = File.ReadAllLines(args[0]);
            Dictionary<string, ModuleItem> module_items = new Dictionary<string, ModuleItem>();
            foreach (string user in user_list)
            {
                if (File.Exists(user.Trim() + ".stat"))
                {
                    string[] stats = File.ReadAllLines(user.Trim() + ".stat");
                    Console.WriteLine(user.Trim() + ".stat");
                    if (stats.Length > 0)
                    {
                        foreach (string line in stats.Skip(1))
                        {
                            string[] segs = line.Split('\t');
                            if (segs.Length >= 6)
                            {
                                string module_name = segs[0];
                                string family_id = segs[1];
                                string type = segs[2];
                                int total_use = int.Parse(segs[3]);
                                int total_run = int.Parse(segs[4]);
                                int total_time = int.Parse(segs[5]);

                                if (!module_items.ContainsKey(family_id))
                                {
                                    ModuleItem item = new ModuleItem();
                                    item.Name = module_name;
                                    item.FamilyId = family_id;
                                    item.ModuleType = type;
                                    item.TotalUsed = total_use;
                                    item.TotalRun = total_run;
                                    item.TotalRunTimeSeconds = total_time;

                                    module_items.Add(family_id, item);
                                }
                                else
                                {
                                    module_items[family_id].TotalUsed += total_use;
                                    module_items[family_id].TotalRun += total_run;
                                    module_items[family_id].TotalRunTimeSeconds += total_time;
                                }
                            }
                        }
                    }
                }
            }

            StreamWriter sw = new StreamWriter(args[1]);
            sw.WriteLine(String.Join("\t", new string[]
            {
                "Module Name",
                "Family Id",
                "Module Type",
                "Total Usage",
                "Total Run",
                "Total Runtime (sec)"
            }));

            foreach (string family_id in module_items.Keys)
            {
                ModuleItem print_item = module_items[family_id];
                sw.WriteLine(String.Join("\t", new string[]
                {
                    print_item.Name,
                    print_item.FamilyId,
                    print_item.ModuleType,
                    print_item.TotalUsed.ToString(),
                    print_item.TotalRun.ToString(),
                    print_item.TotalRunTimeSeconds.ToString()
                }));
            }
            sw.Close();
        }
    }
}
