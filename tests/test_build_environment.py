import os
import sys
import unittest
from unittest import mock

import build_exe


class BuildEnvironmentTests(unittest.TestCase):
    def test_project_env_is_accepted(self):
        with mock.patch.object(sys, 'prefix', os.path.join(build_exe.PROJECT_ROOT, 'env')):
            build_exe.check_project_environment()

    def test_global_python_is_rejected_before_cleanup(self):
        with mock.patch.object(sys, 'prefix', os.path.join(build_exe.PROJECT_ROOT, 'other')), \
                mock.patch.object(build_exe.shutil, 'rmtree') as cleanup:
            with self.assertRaisesRegex(SystemExit, 'Builds must use the project env'):
                build_exe.build()
            cleanup.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows DLL search path')
    def test_external_tool_paths_are_removed(self):
        with mock.patch.dict(os.environ, {'PATH': r'C:\external\poppler\bin'}):
            build_exe.configure_build_path()
            paths = os.environ['PATH'].split(os.pathsep)
            self.assertNotIn(r'C:\external\poppler\bin', paths)
            self.assertEqual(paths[0], os.path.join(os.environ['SystemRoot'], 'System32'))
            self.assertIn(os.path.join(sys.prefix, 'Scripts'), paths)
            self.assertIn(sys.base_prefix, paths)

    @unittest.skipUnless(os.name == 'nt', 'Windows ICU resolution')
    def test_icu_resolves_to_windows_not_poppler(self):
        from PyInstaller.depend.bindepend import resolve_library_path

        with mock.patch.dict(os.environ, {'PATH': r'C:\external\poppler\bin'}):
            build_exe.configure_build_path()
            resolved = resolve_library_path('icuuc.dll')
            expected = os.path.join(os.environ['SystemRoot'], 'System32', 'icuuc.dll')
            self.assertEqual(os.path.normcase(resolved), os.path.normcase(expected))


if __name__ == '__main__':
    unittest.main()
