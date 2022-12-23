using Microsoft.Aether.DataContracts;
using Microsoft.Aether.Library;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModuleUsageStat
{
    public class ModuleItem
    {
        public string Name { get; set; }
        public string FamilyId { get; set; }
        public string ModuleType { get; set; }
        public long TotalUsed { get; set; } // including reused
        public long TotalRun { get; set; }  // not include reused
        public long TotalRunTimeSeconds { get; set; }
    }

    internal class Program
    {
        class ExperimentAnalyzer
        {
            string[] users_;
            Dictionary<string, ModuleItem> modules_;
            DateTime base_time_;

            public ExperimentAnalyzer(string[] users, int days)
            {
                this.users_ = users;
                modules_= new Dictionary<string, ModuleItem>();
                base_time_ = DateTime.Now.AddDays(-days);
            }

            public void GetAnalyzeExperiments(AetherEnvironment environment, string output_file)
            {
                foreach (string user in users_) 
                { 
                    AnalyzeExperiments(environment, user.Trim());
                }

                StreamWriter sw = new StreamWriter(output_file);
                sw.WriteLine(String.Join("\t", new string[] { "Module Name", "Family Id", "Module Type", "Total Usage", "Total Run", "Total RunTime (sec)" }));
                foreach (string module in modules_.Keys)
                {
                    sw.WriteLine(String.Join("\t", new string[]
                    {
                        modules_[module].Name,
                        modules_[module].FamilyId,
                        modules_[module].ModuleType,
                        modules_[module].TotalUsed.ToString(),
                        modules_[module].TotalRun.ToString(),
                        modules_[module].TotalRunTimeSeconds.ToString()
                    }));
                }
                sw.Close();
            }

            void AnalyzeExperiments(AetherEnvironment environment, string user)
            {
                string query = "owner:" + user;
                IAetherSearchEntityResponse<IAetherSearchExperiment> searchResult = environment.ExperimentSearch(query + " count:500");
                //IAetherSearchEntityResponse<IAetherSearchExperiment> searchResult = environment.ExperimentSearch("9566c691-a814-4388-9507-c7697b6d202f");

                Console.WriteLine(user);
                Console.WriteLine("------------------");

                StreamWriter sw = new StreamWriter(user + ".log");

                List<string> node_ids= new List<string>();

                int expcnt = 0;
                bool stop = false;
                while (searchResult.Entities.Count() > 0)
                {
                    // TODO: first check if an experiment completed.

                    foreach (var entity in searchResult.Entities)
                    {
                        DateTime? startdate = entity.DateStart;
                        if (startdate != null && startdate < base_time_)
                        {
                            stop = true;
                            break;
                        }
                        
                        sw.WriteLine("running exp: " + entity.Id);

                        IExecutionGraph ex_graph = null;
                        try
                        {
                            ex_graph = entity.GetExecutionGraph();
                        }
                        catch
                        {
                            ex_graph = null;
                        }

                        if (ex_graph != null)
                        {
                            traverse_graph(sw, environment, entity.Id, node_ids, ex_graph, "");
                        }
                    }


                    if (stop)
                    {
                        break;
                    }

                    expcnt += searchResult.Entities.Count();
                    searchResult = environment.ExperimentSearch(query + " count:500 skip:" + expcnt);
                }

                Console.WriteLine();

                sw.Close();
            }

            long GetExecutionTime(AetherEnvironment environment, string exp_id, List<string> node_id_path, string node_id)
            {
                node_id_path.Add(node_id);
                Microsoft.Aether.DataContracts.TaskStatus node_status = null;
                try
                {
                    node_status = environment.GetNodeStatus(exp_id, node_id_path);
                }
                catch
                {
                    Console.WriteLine("fail get statuss: " + String.Join(">", node_id_path));
                    node_status = null;
                }

                node_id_path.RemoveAt(node_id_path.Count - 1);

                // return 0 if it is reused, or start/end time is not available.
                if (node_status == null ||
                    node_status.ReuseInfo != null || 
                    node_status.EndTime == null || 
                    node_status.StartTime == null)
                {
                    return 0;
                } else
                {
                    TimeSpan ts = (TimeSpan)(node_status.EndTime - node_status.StartTime);
                    return (long)(ts.TotalSeconds);
                }
            }

            string ModuleNodeInfo(IModuleNode n, string indent, long total_seconds)
            {
                return indent +
                    String.Join(" - ", new string[] {
                        n.Name,
                        n.Id,
                        //n.Resource.EntityStatus.ToString(), 
                        n.Resource.Owner,
                        n.Resource.ResourceTypeName,
                        //n.Module.Name,
                        //n.Module.Description,
                        n.Module.Version,
                        n.Module.FamilyId,
                        //n.Module.EntityStatus.ToString(),
                        //n.Module.ExecutionEnvironment.ToString(),
                        n.Module.ModuleExecutionType,
                        total_seconds.ToString()
                    });                
            }

            string GraphNodeInfo(ISubGraphNode n, string indent, long total_seconds)
            {
                return indent +
                    String.Join(" - ", new string[] {
                        n.Name,
                        n.Resource.EntityStatus.ToString(),
                        n.Resource.Owner,
                        n.Resource.ResourceTypeName,
                        total_seconds.ToString()
                    });
            }

            void traverse_graph(StreamWriter sw, AetherEnvironment environment, string exp_id, List<string> parents, IGraph graph, string indent)
            {
                IEnumerable<IModuleNode> module_nodes = graph.ModuleNodes;
                IEnumerable<ISubGraphNode> subgraph_nodes = graph.SubGraphNodes;


                sw.WriteLine(indent + "> module nodes");
                foreach (IModuleNode n in module_nodes)
                {
                    string family_id = n.Module.FamilyId;
                    long seconds = GetExecutionTime(environment, exp_id, parents, n.Id);

                    if (!modules_.ContainsKey(family_id))
                    {
                        ModuleItem new_item = new ModuleItem();
                        new_item.Name = n.Name;
                        new_item.FamilyId= family_id;
                        new_item.ModuleType = n.Module.ModuleExecutionType;
                        new_item.TotalUsed = 0;
                        new_item.TotalRunTimeSeconds = 0;
                        new_item.TotalRun = 0;

                        modules_.Add(family_id, new_item);
                    }

                    ModuleItem item = modules_[family_id];
                    item.TotalUsed++;
                    if (seconds > 0)
                    {
                        item.TotalRun++;
                        item.TotalRunTimeSeconds += seconds;
                    }

                    string module_info = ModuleNodeInfo(n, indent, seconds);
                    sw.WriteLine(module_info);
                }

                sw.WriteLine(indent + "> subgraph nodes");
                foreach (ISubGraphNode n in subgraph_nodes)
                {
                    long seconds = GetExecutionTime(environment, exp_id, parents, n.Id);
                    string graph_info = GraphNodeInfo(n, indent, seconds);
                    sw.WriteLine(graph_info);
                    // skip re-used subgraphs
                    if (seconds > 0)
                    {
                        parents.Add(n.Id);
                        traverse_graph(sw, environment, exp_id, parents, n.SubGraph.GetGraph(), indent + "  ");
                        parents.RemoveAt(parents.Count - 1);
                    }
                }
            }
        }
        static void Main(string[] args)
        {
            if (args.Length < 4)
            {
                Console.WriteLine("ModuleUsageStat.exe <user or list> <user list file> <days> <stat file>");
            } 
            else
            {
                string[] users = string.Equals(args[0], "list") ? 
                                    File.ReadAllLines(args[1]) : 
                                    new string[] { args[1] };

                var environment = new AetherEnvironment(endpointAddress: AetherEnvironment.Aether1PEndpointAddress, clientName: "ModuleUsageStat");

                ExperimentAnalyzer analyzer = new ExperimentAnalyzer(users, int.Parse(args[2]));
                analyzer.GetAnalyzeExperiments(environment, args[3]);
            }
        }

    }
}
