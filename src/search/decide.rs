use crate::config::RrfMode;
use crate::error::AppError;
use crate::es::{select_rrf_syntax, ClusterVersion};
use crate::model::{FusionMode, RrfSyntax};

#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalChoice {
    Bm25 { degraded: Option<String> },
    Knn,
    Weighted,
    Native { syntax: RrfSyntax },
    App,
}

pub fn decide(
    configured: RrfMode,
    explicit: Option<FusionMode>,
    version: Option<&ClusterVersion>,
) -> Result<RetrievalChoice, AppError> {
    match explicit {
        Some(FusionMode::Bm25) => Ok(RetrievalChoice::Bm25 { degraded: None }),
        Some(FusionMode::Knn) => {
            reject_old_cluster(version)?;
            Ok(RetrievalChoice::Knn)
        }
        Some(FusionMode::Weighted) => {
            reject_old_cluster(version)?;
            Ok(RetrievalChoice::Weighted)
        }
        Some(FusionMode::AppRrf) => {
            reject_old_cluster(version)?;
            Ok(RetrievalChoice::App)
        }
        Some(FusionMode::NativeRrf) => Ok(RetrievalChoice::Native {
            syntax: native_syntax(configured, version)?,
        }),
        None => default_choice(configured, version),
    }
}

fn default_choice(
    configured: RrfMode,
    version: Option<&ClusterVersion>,
) -> Result<RetrievalChoice, AppError> {
    match configured {
        RrfMode::Retriever => Ok(RetrievalChoice::Native {
            syntax: RrfSyntax::Retriever,
        }),
        RrfMode::SubSearches => Ok(RetrievalChoice::Native {
            syntax: RrfSyntax::SubSearches,
        }),
        RrfMode::App => Ok(RetrievalChoice::App),
        RrfMode::Auto => {
            let version =
                version.ok_or_else(|| AppError::Internal("auto 模式缺少集群版本".into()))?;
            Ok(match select_rrf_syntax(version) {
                Some(RrfSyntax::Retriever) => RetrievalChoice::Native {
                    syntax: RrfSyntax::Retriever,
                },
                Some(RrfSyntax::SubSearches) => RetrievalChoice::Native {
                    syntax: RrfSyntax::SubSearches,
                },
                Some(RrfSyntax::Application) => RetrievalChoice::App,
                None => RetrievalChoice::Bm25 {
                    degraded: Some(format!(
                        "集群 {} 不支持 dense_vector，已退回 BM25",
                        version.raw
                    )),
                },
            })
        }
    }
}

fn native_syntax(
    configured: RrfMode,
    version: Option<&ClusterVersion>,
) -> Result<RrfSyntax, AppError> {
    reject_old_cluster(version)?;
    match configured {
        RrfMode::Retriever => Ok(RrfSyntax::Retriever),
        RrfMode::SubSearches => Ok(RrfSyntax::SubSearches),
        RrfMode::App => Err(AppError::BadRequest(
            "RRF_SYNTAX=app 时请使用 fusion=app_rrf".into(),
        )),
        RrfMode::Auto => {
            let version =
                version.ok_or_else(|| AppError::Internal("auto 模式缺少集群版本".into()))?;
            match select_rrf_syntax(version) {
                Some(RrfSyntax::Retriever) => Ok(RrfSyntax::Retriever),
                Some(RrfSyntax::SubSearches) => Ok(RrfSyntax::SubSearches),
                Some(RrfSyntax::Application) | None => Err(AppError::BadRequest(format!(
                    "集群 {} 没有原生 RRF，请改用 fusion=app_rrf",
                    version.raw
                ))),
            }
        }
    }
}

fn reject_old_cluster(version: Option<&ClusterVersion>) -> Result<(), AppError> {
    if let Some(version) = version {
        if !version.supports_knn() {
            return Err(AppError::BadRequest(format!(
                "集群 {} 不支持 dense_vector，无法走向量或混合检索",
                version.raw
            )));
        }
    }
    Ok(())
}

pub fn allows_bm25_fallback(explicit: Option<FusionMode>) -> bool {
    !matches!(explicit, Some(FusionMode::Bm25 | FusionMode::Knn))
}

#[cfg(test)]
mod tests {
    use super::{decide, RetrievalChoice};
    use crate::config::RrfMode;
    use crate::es::ClusterVersion;
    use crate::model::{FusionMode, RrfSyntax};

    fn version(raw: &str) -> ClusterVersion {
        ClusterVersion::parse(raw).unwrap()
    }

    #[test]
    fn auto_follows_cluster_capability() {
        let choice = |raw: &str| decide(RrfMode::Auto, None, Some(&version(raw))).unwrap();
        assert_eq!(
            choice("8.15.0"),
            RetrievalChoice::Native {
                syntax: RrfSyntax::SubSearches
            }
        );
        assert_eq!(
            choice("8.16.0"),
            RetrievalChoice::Native {
                syntax: RrfSyntax::Retriever
            }
        );
        assert_eq!(choice("8.4.0"), RetrievalChoice::App);
        match choice("7.17.18") {
            RetrievalChoice::Bm25 { degraded: Some(_) } => {}
            other => panic!("expected degraded bm25, got {other:?}"),
        }
    }

    #[test]
    fn explicit_request_overrides_the_default_path() {
        let native = decide(RrfMode::Retriever, Some(FusionMode::NativeRrf), None).unwrap();
        assert_eq!(
            native,
            RetrievalChoice::Native {
                syntax: RrfSyntax::Retriever
            }
        );
        assert_eq!(
            decide(
                RrfMode::Auto,
                Some(FusionMode::AppRrf),
                Some(&version("8.15.0"))
            )
            .unwrap(),
            RetrievalChoice::App
        );
        assert!(decide(
            RrfMode::Auto,
            Some(FusionMode::NativeRrf),
            Some(&version("8.4.0"))
        )
        .is_err());
        assert!(decide(
            RrfMode::Auto,
            Some(FusionMode::Knn),
            Some(&version("7.17.0"))
        )
        .is_err());
    }
}
