using System;
using System.Collections.Generic;
using System.Net;
using System.Text;
using System.Runtime.Serialization;
using Microsoft.Http;
using Microsoft.Search.Polaris.Starlite.Contract;
using System.Linq;
using LegoPortal.Data.Polaris;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using HttpClient = Microsoft.Http.HttpClient;
using HttpContent = Microsoft.Http.HttpContent;


namespace PolarisClient
{
    public static class PolarisV2Proxy
    {
        private const string LEGOPORTAL_POLARIS_API_PATH = "/api/polaris";

        static HttpClient m_httpClient;
        public static void Init(string host)
        {
            m_httpClient = new HttpClient
            {
                TransportSettings =
                {
                    Cookies = new CookieContainer(),
                    Credentials = CredentialCache.DefaultNetworkCredentials,
                    Proxy = null
                }
            };
            m_httpClient.BaseAddress = new Uri(host);
        }

        public static void QueryJobWithName(string name)
        {
            var queryString = Uri.EscapeDataString($"$top=1&$filter=Name eq '{name}'");
            var queryUrl = $"{LEGOPORTAL_POLARIS_API_PATH}?{queryString}";

            Console.WriteLine($"URL: {queryUrl}");
            using (var response = m_httpClient.Get(queryUrl))
            {
                if (response.StatusCode != HttpStatusCode.OK)
                {
                    var errorResponse = (JObject)JsonConvert.DeserializeObject(response.Content.ReadAsString());
                    var errorMsg = errorResponse.GetValue("Message");
                    Console.WriteLine("Failed to create a job. StatusCode = {0}, Error Message {1}",
                        response.StatusCode,
                        errorMsg);
                }
                else
                {
                    var successResponse = (JObject)JsonConvert.DeserializeObject(response.Content.ReadAsString());
                    var targetGuid = (string)successResponse.SelectToken("Items[0].Id");

                    Console.WriteLine($"Fetched Guid: {targetGuid}");
                }
            }
        }

        public static string SubmitJobUsingDTO(string requestBody)
        {
            using (var response = m_httpClient.Post(LEGOPORTAL_POLARIS_API_PATH, HttpContent.Create(requestBody, Encoding.UTF8, "application/json")))
            {

                if (response.StatusCode != HttpStatusCode.Created)
                {
                    var errorMsg = PolarisV2Proxy.GetErrorMessage(response,
                        "Failed to create a job. StatusCode = {0}, Error Message: \n{1}");
                    Console.Error.WriteLine(errorMsg);
                    throw new ApplicationException(errorMsg);
                }

                var responseText = response.Content.ReadAsString();
                var obj = (JObject)JsonConvert.DeserializeObject(responseText);
                return obj.GetValue("Id").ToString();

            }
        }

        public static string FetchJobResult(string guid)
        {
            using (var response = m_httpClient.Get($"{LEGOPORTAL_POLARIS_API_PATH}/{guid}"))
            {
                if (response.StatusCode != HttpStatusCode.OK)
                {

                    var errorMsg = PolarisV2Proxy.GetErrorMessage(response,
                        "Failed to fetch a job. StatusCode = {0}, Error Message: \n{1}");
                    Console.Error.WriteLine(errorMsg);
                    throw new ApplicationException(errorMsg);
                }
                return response.Content.ReadAsString();
            }
        }

        private static string GetErrorMessage(HttpResponseMessage response, string template)
        {
            string errorResponseText = null;
            try
            {
                errorResponseText = response.Content.ReadAsString();
            }
            catch (Exception)
            {

            }
            return string.Format(template,
                response.StatusCode,
                errorResponseText ?? "None");
        }

        internal static string GetTestLogsLocation(JObject statusObject)
        {
            JArray testRuns;
            try
            {
                testRuns = (JArray)statusObject.GetValue("Tests")[0]["TestRuns"];
            }
            catch (NullReferenceException)
            {
                // Node is missing from the JSON.
                Console.Error.WriteLine("Failed to find expected .Tests[0].TestRuns in JSON blob:");
                Console.Error.WriteLine(statusObject.ToString());
                return string.Empty;
            }

            if (testRuns.Count == 0)
            {
                // Failure before attempted test run.
                return string.Empty;
            }

            JToken resultLocation = testRuns[0]["TestResultsLocation"];
            if (resultLocation == null)
            {
                return string.Empty;
            }

            return resultLocation.ToString();
        }
    }
}
