This module is used to get impressions corresponding to specified flight from Bing Logs. It has 4 parameters：

- FlightId: used to filter some flights you need, you can use 'ir.+' to collect all flights. 
- StartTime：the start time to collect IL data
- EndTime: the end time to collect IL data
- HeavyDebugger: bool varibale. "False" means we collect IL data from XTCache, it can only output limitted columns and runs very fast. "True" means we collect data from MSLAPI, it can output extra debugger columns, but runs much slower. If you want to further debug IL, please select it as "True."

Detail information about IL debugger/Inline metrics, please refer this [link](https://microsoftapc-my.sharepoint.com/:o:/r/personal/binxjia_microsoft_com/_layouts/15/Doc.aspx?sourcedoc=%7B64aac550-a171-48d6-a496-97c7d2bd2365%7D&action=edit&wd=target(Relevance%2FKnowledge.one%7C2DC9FBCA-5ED9-4202-9B0B-71B07CE4FC3C%2FInterleaving%20Debugger%5C%2FInline%20Metrics%7C97D3FA0B-6D93-40A7-9A10-D0CC24BEC518%2F)&share=IgFQxapkcaHWSKSWl8fSvSNlATGkst_axaYZYoVNyBXqsSQ).
