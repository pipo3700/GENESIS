using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

namespace Function1_Upload
{
    public static class UploadCVandOffer
    {
        [FunctionName("UploadCVandOffer")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", "options", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("Processing files upload...");

            try
            {
                var form = await req.ReadFormAsync();

                var cvFile = form.Files["cv"];
                var jobOfferText = form["jobOffer"];

                if (cvFile == null || string.IsNullOrEmpty(jobOfferText))
                    return new BadRequestObjectResult("Faltan el archivo del CV o la descripción de la oferta de trabajo.");

                var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
                var blobServiceClient = new BlobServiceClient(connectionString);
                var containerClient = blobServiceClient.GetBlobContainerClient("upload");

                await containerClient.CreateIfNotExistsAsync(PublicAccessType.None);

                var timestamp = DateTime.UtcNow.Ticks;
                var jobId = timestamp.ToString(); 

                // Subir CV a 'cv/' dentro del contenedor
                var cvBlobName = $"cv/cv-{timestamp}-{cvFile.FileName}";
                var cvBlobClient = containerClient.GetBlobClient(cvBlobName);

                using (var stream = cvFile.OpenReadStream())
                {
                    await cvBlobClient.UploadAsync(stream, true);
                }

                // Subir descripción de oferta como .txt en 'joboffer/'
                var jobBlobName = $"joboffer/jobOffer-{timestamp}.txt";
                var jobBlobClient = containerClient.GetBlobClient(jobBlobName);

                using (var ms = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(jobOfferText)))
                {
                    await jobBlobClient.UploadAsync(ms, true);
                }

                return new OkObjectResult(new
                {
                    message = "Subida exitosa.",
                    jobId = jobId,
                    cvUrl = cvBlobClient.Uri.ToString(),
                    jobOfferUrl = jobBlobClient.Uri.ToString()
                });
            }
            catch (Exception ex)
            {
                log.LogError($"Error al subir archivos: {ex.Message}");
                return new StatusCodeResult(StatusCodes.Status500InternalServerError);
            }
        }
        [FunctionName("Hello")]
        public static IActionResult Hello(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get")] HttpRequest req,
            ILogger log)
        {
            return new OkObjectResult("Hola mundo desde Azure Functions!");
        }
    }
}
