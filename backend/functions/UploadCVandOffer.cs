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

namespace functions
{
    public static class UploadCVandOffer
    {
        [FunctionName("UploadCVandOffer")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("Procesando subida de archivos...");

            try
            {
                var form = await req.ReadFormAsync();

                var cvFile = form.Files["cv"];
                var jobOfferText = form["jobOffer"];

                if (cvFile == null || string.IsNullOrEmpty(jobOfferText))
                    return new BadRequestObjectResult("Faltan el archivo del CV o la descripción de la oferta de trabajo.");

                // Cliente de Blob Storage
                var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
                var blobServiceClient = new BlobServiceClient(connectionString);
                var containerClient = blobServiceClient.GetBlobContainerClient("uploads");

                await containerClient.CreateIfNotExistsAsync(PublicAccessType.None);

                // Subir CV
                var cvBlobName = $"cv-{DateTime.UtcNow.Ticks}-{cvFile.FileName}";
                var cvBlobClient = containerClient.GetBlobClient(cvBlobName);

                using (var stream = cvFile.OpenReadStream())
                {
                    await cvBlobClient.UploadAsync(stream, true);
                }

                // Subir descripción de oferta como .txt
                var jobBlobName = $"jobOffer-{DateTime.UtcNow.Ticks}.txt";
                var jobBlobClient = containerClient.GetBlobClient(jobBlobName);

                using (var ms = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(jobOfferText)))
                {
                    await jobBlobClient.UploadAsync(ms, true);
                }

                return new OkObjectResult(new
                {
                    message = "Subida exitosa.",
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
    }
}
