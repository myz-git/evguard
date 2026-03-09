# fingerprint.py 生成机器指纹
import hashlib
import winreg
import pyperclip

def get_machine_guid() -> str:
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
    guid, _ = winreg.QueryValueEx(key, "MachineGuid")
    return str(guid).strip()

def machine_fingerprint() -> str:
    guid = get_machine_guid().encode("utf-8")
    return hashlib.sha256(guid).hexdigest()

if __name__ == "__main__":
    fp = machine_fingerprint()
    print("RequestCode (Machine Fingerprint):")
    print(fp)
    try:
        pyperclip.copy(fp)
        print("\n已复制到剪贴板：直接粘贴发给我即可。")
    except Exception:
        print("\n复制剪贴板失败：请手动复制上面的指纹发给我。")