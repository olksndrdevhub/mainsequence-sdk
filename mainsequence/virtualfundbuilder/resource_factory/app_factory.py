import hashlib
import json
import os
from abc import abstractmethod

from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.enums import ResourceType
from mainsequence.virtualfundbuilder.resource_factory.base_factory import insert_in_registry, BaseResource
from mainsequence.virtualfundbuilder.utils import get_vfb_logger

logger = get_vfb_logger()

class BaseApp(BaseResource):
    TYPE = ResourceType.APP

APP_REGISTRY = APP_REGISTRY if 'APP_REGISTRY' in globals() else {}
def register_app(name=None, register_in_agent=True):
    """
    Decorator to register a model class in the factory.
    If `name` is not provided, the class's name is used as the key.
    """
    def decorator(cls):
        return insert_in_registry(APP_REGISTRY, cls, register_in_agent, name)
    return decorator


class HtmlApp(BaseApp):
    """
    A base class for apps that generate HTML output.
    """
    TYPE = ResourceType.HTML_APP

    def _get_hash_from_configuration(self):
        try:
            return hashlib.sha256(
                json.dumps(self.configuration.__dict__, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"[{self.__name__}] Could not hash configuration: {e}")

    def __init_subclass__(cls, **kwargs):
        """
        Wraps the subclass's `run` method to add validation and saving logic.
        """
        super().__init_subclass__(**kwargs)
        original_run = cls.run

        def run_wrapper(self, *args, **kwargs) -> str:
            # Execute the user-defined run method from the subclass
            html_content = original_run(self, *args, **kwargs)

            if not isinstance(html_content, str):
                raise TypeError(f"The 'run' method of {cls.__name__} must return a string of HTML content.")

            # Store the output
            unique_hash = self._get_hash_from_configuration()
            output_name = f"{cls.__name__}_{unique_hash}.html"
            try:
                with open(output_name, "w", encoding="utf-8") as f:
                    f.write(html_content)

                logger.info(f"[{cls.__name__}] Successfully saved HTML to: {output_name}")
            except IOError as e:
                logger.error(f"[{cls.__name__}] Error saving file: {e}")
                raise

            job_id = os.getenv("JOB_ID", None)
            if job_id:
                html_artifact = None
                try:
                    html_artifact = Artifact.upload_file(
                        filepath=output_name,
                        name=output_name,
                        created_by_resource_name=self.__class__.__name__,
                        bucket_name="HTMLOutput"
                    )
                    logger.info(f"Artifact uploaded successfully: {html_artifact.id if html_artifact else 'Failed'}")
                except Exception as e:
                    logger.info(f"Error uploading artifact: {e}")

        # Replace the subclass's run method with our enhanced wrapper
        cls.run = run_wrapper

    @abstractmethod
    def run(self) -> str:
        """
        This method should be implemented by subclasses to return HTML content as a string.
        The base class will handle saving the output.
        """
        raise NotImplementedError("Subclasses of HtmlApp must implement the 'run' method.")
