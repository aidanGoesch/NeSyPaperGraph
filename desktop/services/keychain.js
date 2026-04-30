const keytar = require("keytar");

async function getSecret(service, account) {
    try {
        return await keytar.getPassword(service, account);
    } catch (error) {
        console.warn("Keychain read failed:", error);
        return "";
    }
}

async function setSecret(service, account, value) {
    await keytar.setPassword(service, account, value);
}

async function deleteSecret(service, account) {
    try {
        await keytar.deletePassword(service, account);
    } catch (error) {
        console.warn("Keychain delete failed:", error);
    }
}

module.exports = {
    getSecret,
    setSecret,
    deleteSecret,
};
