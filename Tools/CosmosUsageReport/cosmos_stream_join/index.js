const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const { Console } = require('console');
const fullPath = path.join(__dirname, 'input')

let crawl = async (browser, username) => {
  const page = await browser.newPage();
  await page.goto(`https://www.cosmos09.osdinfra.net/cosmos/relevance/_Jobs/?userName=${username}@microsoft.com`);
  try {
    await page.waitForSelector('div#EndedJobs tr[data-uid]', { timeout: 5000 });
  } catch (e) {
    console.log(e);
    await page.close();
    return [];
  }

  let res = await page.evaluate(async () => {
    return new Promise((resolve, reject) => {
      let attrs = []
      document.querySelectorAll('div#EndedJobs tr[data-uid]').forEach(e => {
        let tds = e.querySelectorAll('td')
        if (tds.length >= 10) {
          let job_name = tds.item(1).textContent;
          if(job_name.includes("@@@Cosmos_Stream_Join_Common@@@")){
            let run_time = tds.item(9).innerText.split(":");
            let run_time_hours = parseInt(run_time[0]);
            let run_time_minutes = parseInt(run_time[1]);
            let run_time_seconds = parseInt(run_time[2]);
            if (run_time_hours !== 0 || run_time_minutes !== 0 || run_time_seconds !== 0) {
              let run_time_in_hour = run_time_hours + run_time_minutes / 60.0 + run_time_seconds / 3600.0;
              attrs.push({
                tokens: tds.item(5).innerText,
                duration: run_time_in_hour / 24.0,
                jobname: job_name
              })
            }
          }
        }
      })
      resolve(attrs)
    })
  })
  await page.close();
  return res;
};

(async () => {
  let teams = [];
  let files = fs.readdirSync(fullPath);
  files.forEach(file => {
    let teamName = file.split('.')[0]
    let usernames = fs.readFileSync(path.join(fullPath, file), 'utf8').split('\n').map((username) => username.trim())
    teams.push({
      name: teamName, usernames: usernames
    })
  })
  console.log(teams)
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: false,
    defaultViewport: {
      width: 1920,
      height: 1080
    }
  });
  res_lines = []
  total_job_count = 0
  total_tokens = 0
  utilization = 0.0
  for (let team of teams) {
    for (let username of team['usernames']) {
      let res = await crawl(browser, username)
      total_job_count += res.length
      res.forEach((item) => {
        tokens = parseInt(item['tokens'])
        total_tokens += tokens
        utilization += parseFloat(item['duration']) * tokens
      })
    }
  }
  res_lines.push(`${total_job_count}\t${total_tokens}\t${utilization}\n`)
  fs.writeFileSync(`dist/results.tsv`, res_lines.join(''))
  await browser.close();
})();

