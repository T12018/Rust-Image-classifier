use std::path::PathBuf;
use directories::ProjectDirs;
use burn::data::dataset::vision::ImageFolderDataset;

fn data_root() -> PathBuf {
    let data_dir;
    if let Some(proj) = ProjectDirs::from("burn-examples", "", "custom-image-dataset") {
        data_dir = proj.data_dir().to_path_buf();
    } else {
        data_dir = std::env::current_dir().unwrap().join("data");
    }
    println!("Data dir: {:?}", data_dir);
    data_dir
}

// New: skin_cancer dataset accessors
pub trait SkinCancerLoader {
    fn skin_cancer_train() -> Self;
    fn skin_cancer_test() -> Self;
}

impl SkinCancerLoader for ImageFolderDataset {
    fn skin_cancer_train() -> Self {
        let root = data_root().join("skin_cancer");
        ImageFolderDataset::new_classification(root.join("train")).unwrap()
    }

    fn skin_cancer_test() -> Self {
        let root = data_root().join("skin_cancer");
        ImageFolderDataset::new_classification(root.join("test")).unwrap()
    }
}