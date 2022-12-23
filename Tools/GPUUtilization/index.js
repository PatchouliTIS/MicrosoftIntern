const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path')

const fullPath = path.join(__dirname, 'input')

const lead_summary_btn = 'mat-action-list>button:nth-of-type(4)';
const team_usage_btn = 'mat-action-list>button:nth-of-type(2)';
const gpu_utilization_btn = 'mat-action-list>button:nth-of-type(3)';
const lead_tabel = 'visual-modern>div>div>div.tableEx>div.innerContainer';
const binxing_title = 'div[title="binxjia"]';
const idle_hours_prefix = 'visual-container-repeat>visual-container:nth-of-type(9)>transform>div>div.visualContent>div';
const idle_hours_table = idle_hours_prefix + '>visual-modern>div>div>div.tableEx>div.innerContainer';
const top_user_prefix = 'visual-container-repeat>visual-container:nth-of-type(6)>transform>div>div.visualContent>div';
const top_user_table = top_user_prefix + '>visual-modern>div>div>div.tableEx>div.innerContainer';

function delay(n){
  return new Promise(function(resolve){
      setTimeout(resolve,n*1000);
  });
}

let set_date_lead_summary = async () => {
  const prefix = 'visual-container-repeat>visual-container:nth-of-type(8)>transform>div>div.visualContent>div>visual-modern>div>div.slicer-container>div.slicer-content-wrapper'
  const date_path = prefix + '>div.date-slicer>div.date-slicer-head>div.date-slicer-range'
  const date1 = date_path + '>div:nth-of-type(1)>div>input';
  const date2 = date_path + '>div:nth-of-type(2)>div>input';

  const today = new Date();
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const lastweek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  const yesterday_str = `${yesterday.getMonth() + 1}/${yesterday.getDate()}/${yesterday.getFullYear()}`
  const lastweek_str = `${lastweek.getMonth() + 1}/${lastweek.getDate()}/${lastweek.getFullYear()}`

  console.log(yesterday_str)
  console.log(lastweek_str)

  return new Promise((resolve, reject) => {
    let attrs = [];

    document.querySelector(date2).value = yesterday_str;
    document.querySelector(date2).dispatchEvent(new Event('change'));
    document.querySelector(date1).value = lastweek_str;
    document.querySelector(date1).dispatchEvent(new Event('change'));

    resolve(attrs)
  })
};


let set_date_top_user = async () => {
  const prefix = 'visual-container-repeat>visual-container:nth-of-type(2)>transform>div>div.visualContent>div>visual-modern>div>div.slicer-container>div.slicer-content-wrapper'
  const date_path = prefix + '>div.date-slicer>div.date-slicer-head>div.date-slicer-range'
  const date1 = date_path + '>div:nth-of-type(1)>div>input';
  const date2 = date_path + '>div:nth-of-type(2)>div>input';

  const today = new Date();
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const lastweek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  const yesterday_str = `${yesterday.getMonth() + 1}/${yesterday.getDate()}/${yesterday.getFullYear()}`
  const lastweek_str = `${lastweek.getMonth() + 1}/${lastweek.getDate()}/${lastweek.getFullYear()}`

  console.log(yesterday_str)
  console.log(lastweek_str)

  return new Promise((resolve, reject) => {
    let attrs = [];

    document.querySelector(date2).value = yesterday_str;
    document.querySelector(date2).dispatchEvent(new Event('change'));
    document.querySelector(date1).value = lastweek_str;
    document.querySelector(date1).dispatchEvent(new Event('change'));

    resolve(attrs)
  })
};

let lead_summary_shot = async (browser) => {


  const page = await browser.newPage();
  await page.goto(`https://msit.powerbi.com/groups/4b050379-eee2-4d14-bd89-269a107097d3/reports/98f0c108-540c-4c98-9b11-0efbd4bcc01d/ReportSection`);
  try {
    await page.waitForSelector(lead_summary_btn, { timeout: 20000 });
  } catch (e) {
    console.log(e);
    await page.close();
    return [];
  }

  // lead summary
  const lsb = await page.$(lead_summary_btn);
  await lsb.click();

  await delay(2);

  try {
    await page.waitForSelector(lead_tabel, { timeout: 10000 });
  } catch (e) {
    console.log(e);
    await page.close();
    return [];
  }

  const res = await page.evaluate(set_date_lead_summary);
  await delay(3);

  const lt = await page.$(lead_tabel);
  await lt.screenshot({ path: 'lead_summary.png' });

  await delay(2);

  await page.close();
};

let top_users_shot = async (browser) => {
  const page = await browser.newPage();
  await page.goto(`https://msit.powerbi.com/groups/4b050379-eee2-4d14-bd89-269a107097d3/reports/98f0c108-540c-4c98-9b11-0efbd4bcc01d/ReportSection`);
  try {
    await page.waitForSelector(lead_summary_btn, { timeout: 20000 });
  } catch (e) {
    console.log(e);
    await page.close();
    return [];
  }

  // top users
  const tub = await page.$(team_usage_btn);
  await tub.click();

  await delay(2);

  try {
    await page.waitForSelector(binxing_title, { timeout: 10000 });
  } catch (e) {
    console.log(e);
    await page.close();
    return [];
  }

  await page.evaluate(set_date_top_user);
  await delay(3);

  const bt = await page.$(binxing_title);
  await bt.click();

  await delay(2);

  const lt2 = await page.$(lead_tabel);
  await lt2.screenshot({ path: 'top_users.png' });

  await delay(2);

  await page.close();
};

let hihg_idle_hours_shot = async (browser) => {

  const page = await browser.newPage();
  await page.goto(`https://msit.powerbi.com/groups/4b050379-eee2-4d14-bd89-269a107097d3/reports/98f0c108-540c-4c98-9b11-0efbd4bcc01d/ReportSection`);
  try {
    await page.waitForSelector(lead_summary_btn, { timeout: 20000 });
  } catch (e) {
    console.log(e);
    await page.close();
    return [];
  }

  // high idle hours
  const gub = await page.$(gpu_utilization_btn);
  await gub.click();

  await delay(2);

  await page.evaluate(set_date_top_user);
  await delay(3);

  try {
    await page.waitForSelector(binxing_title, { timeout: 10000 });
  } catch (e) {
    console.log(e);
    await page.close();
    return [];
  }

  const bt2 = await page.$(binxing_title);
  await bt2.click();

  await delay(1);

  const lt3 = await page.$(idle_hours_table);
  await lt3.screenshot({ path: 'high_idle_hours.png' });

  await delay(2);
  
  await page.close();
};

(async () => {

  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: false,
    defaultViewport: {
      width: 2560,
      height: 1440
    }
  });

  await lead_summary_shot(browser);
  await top_users_shot(browser);
  await hihg_idle_hours_shot(browser);
  await browser.close();
})();

