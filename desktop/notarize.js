const { notarize } = require("@electron/notarize");

module.exports = async function notarizeBuild(context) {
    const { electronPlatformName, appOutDir, packager } = context;
    if (electronPlatformName !== "darwin") return;
    if (!process.env.APPLE_ID || !process.env.APPLE_APP_SPECIFIC_PASSWORD) {
        return;
    }

    const appName = packager.appInfo.productFilename;
    await notarize({
        appPath: `${appOutDir}/${appName}.app`,
        appleId: process.env.APPLE_ID,
        appleIdPassword: process.env.APPLE_APP_SPECIFIC_PASSWORD,
        teamId: process.env.APPLE_TEAM_ID,
    });
};
