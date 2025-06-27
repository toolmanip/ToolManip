from github import Github
import base64
import os
import time

def upload_image_to_github(local_img_path, repo_name, github_token, target_path_in_repo, commit_message="upload image"):
    g = Github(github_token)
    user = g.get_user()
    repo = g.get_repo(repo_name)

    with open(local_img_path, "rb") as f:
        content = f.read()
        encoded_content = base64.b64encode(content).decode("utf-8")

    file_name = os.path.basename(local_img_path)
    target_path = f"{target_path_in_repo}/{file_name}"

    try:
        existing_file = repo.get_contents(target_path)
        repo.update_file(
            path=target_path,
            message=commit_message,
            content=content,
            sha=existing_file.sha
        )
        print(f"✅ Updated existing file: {target_path}")
    except:
        repo.create_file(
            path=target_path,
            message=commit_message,
            content=content
        )
        print(f"✅ Uploaded new file: {target_path}")

    raw_url = f"https://raw.githubusercontent.com/{repo_name}/main/{target_path}"
    return raw_url


def RequireURL(imgPath):
    import uuid
    GITHUB_TOKEN = "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"  
    NowTime = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    REPO_NAME = "qizhou98/URL_Img"  
    
    TARGET_PATH_IN_REPO = f"images/{NowTime}"  
    url = upload_image_to_github(imgPath, REPO_NAME, GITHUB_TOKEN, TARGET_PATH_IN_REPO)
    return url


# 示例调用
if __name__ == "__main__":
    GITHUB_TOKEN = "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"
    REPO_NAME = "qizhou98/URL_Img"
    LOCAL_IMAGE_PATH = "/home/guyuwei/SAM-6D-New/D435/rgb_test.png" 
    TARGET_PATH_IN_REPO = "images"

    url = upload_image_to_github(LOCAL_IMAGE_PATH, REPO_NAME, GITHUB_TOKEN, TARGET_PATH_IN_REPO)
    print("Raw URL:", url)
